//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "ParsedJITTensor.h"
#include "libmesh/extrasrc/fptypes.hh"

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/type_hashing.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>

#include <torch/csrc/autograd/grad_mode.h>

#include <ATen/TensorOperators.h>

ParsedJITTensor::ParsedJITTensor() : FunctionParserAD(), _data(*getParserData()) {}

void
ParsedJITTensor::setupTensors()
{
  using namespace torch::jit;

  // allocate node stack
  std::vector<Value *> s(_data.mStackSize);

  // create graph
  _graph = std::make_shared<Graph>();

  // convert immediate data
  _constant_immed.clear();
  for (const auto & immed : _data.mImmed)
    _constant_immed.push_back(_graph->insertConstant(immed));

  // create input nodes
  _input.clear();
  for (unsigned i = 0; i < _data.mVariablesAmount; ++i)
    _input.push_back(_graph->addInput());

  for (const auto & i : _input)
    std::cout << "Input requires grad = " << i->requires_grad() << '\n';

  // create output node
  // _output = _graph->addOutput();

  // build graph
  using namespace FUNCTIONPARSERTYPES;

  // get a reference to the stored bytecode
  const auto & ByteCode = _data.mByteCode;

  int nImmed = 0, sp = -1, op;
  for (unsigned int i = 0; i < ByteCode.size(); ++i)
  {
    // execute bytecode
    switch (op = ByteCode[i])
    {
      case cImmed:
        ++sp;
        std::cout << "Added immed " << _data.mImmed[nImmed] << '\n';
        s[sp] = _constant_immed[nImmed++];
        break;
      case cAdd:
        --sp;
        s[sp] = _graph->insert(aten::add, {s[sp], s[sp + 1]});
        break;
      case cSub:
        --sp;
        s[sp] = _graph->insert(aten::sub, {s[sp], s[sp + 1]});
        break;
      case cRSub:
        --sp;
        s[sp] = _graph->insert(aten::sub, {s[sp + 1], s[sp]});
        break;
      case cMul:
        --sp;
        s[sp] = _graph->insert(aten::mul, {s[sp], s[sp + 1]});
        break;
      case cDiv:
        --sp;
        s[sp] = _graph->insert(aten::div, {s[sp], s[sp + 1]});
        break;

      default:
        if (op >= VarBegin)
        {
          // // load variable
          ++sp;
          s[sp] = _input[op - VarBegin];
        }
        else
        {
          throw std::runtime_error("Opcode not supported for libtorch tensors.");
        }
    }
  }

  auto outputs = s[sp]->node()->outputs();
  for (auto output : outputs)
    _graph->registerOutput(output);

  // make sure graph is well formed
  _graph->lint();

  // optimization
  EliminateDeadCode(_graph); // Tracing of some ops depends on the DCE trick
  ConstantPropagation(_graph);
  EliminateCommonSubexpression(_graph);
  FuseGraph(_graph, true);
}

namespace
{
template <class... Inputs>
inline std::vector<c10::IValue>
makeStack(Inputs &&... inputs)
{
  return {std::forward<Inputs>(inputs)...};
}
}

neml2::Scalar
ParsedJITTensor::Eval(at::ArrayRef<at::Tensor> params)
{
  using namespace torch::jit;

  // copy stuff around (actshually... this should reference the same storage )
  Stack stack;
  for (const auto & i : params)
    stack.push_back(i);

  if (_input.size() != params.size())
    throw std::runtime_error("Unexpected number of inputs in ParsedJITTensor::Eval.");

  // disable autograd
  const auto ad = torch::autograd::GradMode::is_enabled();
  torch::autograd::GradMode::set_enabled(false);

  GraphExecutor exec(_graph, "F");
  auto plan = exec.getPlanFor(stack, 10);
  InterpreterState(plan.code).run(stack);

  print();

  // exec.run(stack);
  if (stack.size() != 1)
    throw std::runtime_error("Unexpected number vof outputs in ParsedJITTensor::Eval.");

  torch::autograd::GradMode::set_enabled(ad);

  return stack[0].toTensor();
}
