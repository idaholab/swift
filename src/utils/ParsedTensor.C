/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ParsedTensor.h"

ParsedTensor::ParsedTensor()
{
}

bool ParsedTensor::parse(const std::string & expression, const std::vector<std::string> & variables)
{
  _variables = variables;
  _error.clear();

  _ast = _parser.parse(expression);
  if (!_ast)
  {
    _error = _parser.errorMessage();
    return false;
  }

  return true;
}

void ParsedTensor::differentiate(const std::string & var)
{
  if (!_ast)
  {
    _error = "No expression to differentiate";
    return;
  }

  _ast = _ast->differentiate(var);
}

void ParsedTensor::optimize()
{
  if (!_ast)
  {
    _error = "No expression to optimize";
    return;
  }

  _ast = _ast->simplify();
}

torch::Tensor ParsedTensor::eval(const std::vector<const torch::Tensor *> & params)
{
  if (!_ast)
    throw std::runtime_error("No expression to evaluate");

  if (params.size() != _variables.size())
    throw std::runtime_error("Parameter count mismatch");

  // Build a JIT graph
  auto graph = std::make_shared<torch::jit::Graph>();
  std::unordered_map<std::string, torch::jit::Value *> var_map;

  // Create input nodes
  for (size_t i = 0; i < _variables.size(); ++i)
  {
    auto input = graph->addInput();
    var_map[_variables[i]] = input;
  }

  // Build the graph from AST
  auto output = _ast->buildGraph(*graph, var_map);

  // Register outputs
  auto outputs = output->node()->outputs();
  for (auto out : outputs)
    graph->registerOutput(out);

  // Lint and optimize graph
  graph->lint();

  // Apply torch optimizations
  torch::jit::EliminateDeadCode(graph);
  torch::jit::ConstantPropagation(graph);
  torch::jit::EliminateCommonSubexpression(graph);
  torch::jit::FuseGraph(graph, true);

  // Execute the graph
  auto executor = std::make_shared<torch::jit::GraphExecutor>(graph, "F");

  torch::jit::Stack stack;
  for (const auto & p : params)
    stack.push_back(*p);

  torch::NoGradGuard no_grad;
  executor->run(stack);

  if (stack.size() != 1)
    throw std::runtime_error("Unexpected number of outputs");

  return stack[0].toTensor();
}

std::string ParsedTensor::toString() const
{
  if (!_ast)
    return "<no expression>";
  return _ast->toString();
}
