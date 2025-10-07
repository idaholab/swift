/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ParsedJITTensor.h"

#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/autograd/grad_mode.h>

ParsedJITTensor::ParsedJITTensor()
{
}

bool ParsedJITTensor::parse(const std::string & expression, const std::vector<std::string> & variables)
{
  _variables = variables;
  _error.clear();
  _graph.reset();
  _executor.reset();

  _ast = _parser.parse(expression);
  if (!_ast)
  {
    _error = _parser.errorMessage();
    return false;
  }

  return true;
}

void ParsedJITTensor::differentiate(const std::string & var)
{
  if (!_ast)
  {
    _error = "No expression to differentiate";
    return;
  }

  _ast = _ast->differentiate(var);
  _graph.reset();
  _executor.reset();
}

void ParsedJITTensor::compile()
{
  if (!_ast)
  {
    _error = "No expression to compile";
    return;
  }

  // Optimize the AST first
  _ast = _ast->simplify();

  // Build a JIT graph
  _graph = std::make_shared<torch::jit::Graph>();
  std::unordered_map<std::string, torch::jit::Value *> var_map;

  // Create input nodes
  for (const auto & var : _variables)
  {
    auto input = _graph->addInput();
    var_map[var] = input;
  }

  // Build the graph from AST
  auto output = _ast->buildGraph(*_graph, var_map);

  // Register outputs
  auto outputs = output->node()->outputs();
  for (auto out : outputs)
    _graph->registerOutput(out);

  // Lint and optimize graph
  _graph->lint();

  // Apply torch optimizations
  torch::jit::EliminateDeadCode(_graph);
  torch::jit::ConstantPropagation(_graph);
  torch::jit::EliminateCommonSubexpression(_graph);
  torch::jit::FuseGraph(_graph, true);

  // Create executor
  _executor = std::make_shared<torch::jit::GraphExecutor>(_graph, "F");
}

torch::Tensor ParsedJITTensor::eval(const std::vector<const torch::Tensor *> & params)
{
  if (!_ast)
    throw std::runtime_error("No expression to evaluate");

  if (params.size() != _variables.size())
    throw std::runtime_error("Parameter count mismatch");

  // Compile if not already done
  if (!_executor)
    compile();

  // Build stack
  torch::jit::Stack stack;
  for (const auto & p : params)
    stack.push_back(*p);

  // Execute
  torch::NoGradGuard no_grad;
  _executor->run(stack);

  if (stack.size() != 1)
    throw std::runtime_error("Unexpected number of outputs");

  return stack[0].toTensor();
}

void ParsedJITTensor::print() const
{
  if (_graph)
    _graph->dump();
  else
    std::cout << "No compiled graph\n";
}

std::string ParsedJITTensor::toString() const
{
  if (!_ast)
    return "<no expression>";
  return _ast->toString();
}
