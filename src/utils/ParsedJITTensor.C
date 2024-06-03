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

#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

#include <torch/csrc/autograd/grad_mode.h>

#include <ATen/TensorOperators.h>

ParsedJITTensor::ParsedJITTensor()
  : FunctionParserAD(), _graph_executor(nullptr), _execution_plan(nullptr), _data(*getParserData())
{
}

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

  // math constants
  const auto const_log10 = _graph->insertConstant(std::log(10.0));
  const auto const_pi = _graph->insertConstant(libMesh::pi);
  const auto const_minus_one = _graph->insertConstant(-1.0);
  const auto const_minus_one_half = _graph->insertConstant(-0.5);
  const auto const_one_third = _graph->insertConstant(1.0 / 3.0);
  const auto const_two = _graph->insertConstant(2.0);

  // create input nodes
  _input.clear();
  for (unsigned i = 0; i < _data.mVariablesAmount; ++i)
    _input.push_back(_graph->addInput());

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

      case cMod:
        --sp;
        s[sp] = _graph->insert(aten::fmod, {s[sp], s[sp + 1]});
        break;
      case cRDiv:
        --sp;
        s[sp] = _graph->insert(aten::div, {s[sp + 1], s[sp]});
        break;

      case cSin:
        s[sp] = _graph->insert(aten::sin, {s[sp]});
        break;
      case cCos:
        s[sp] = _graph->insert(aten::cos, {s[sp]});
        break;
      case cTan:
        // no native a10 tan operator?
        s[sp] = _graph->insert(
            aten::div, {_graph->insert(aten::sin, {s[sp]}), _graph->insert(aten::cos, {s[sp]})});
        break;

      case cSinCos:
        s[sp + 1] = _graph->insert(aten::cos, {s[sp]});
        s[sp] = _graph->insert(aten::sin, {s[sp]});
        ++sp;
        break;

      case cAbs:
        s[sp] = _graph->insert(aten::abs, {s[sp]});
        break;
      case cMax:
        --sp;
        s[sp] = _graph->insert(aten::max, {s[sp], s[sp + 1]});
        break;
      case cMin:
        --sp;
        s[sp] = _graph->insert(aten::min, {s[sp], s[sp + 1]});
        break;

      case cInt:
        s[sp] = _graph->insert(aten::round, {s[sp]});
        break;

      case cLog:
        s[sp] = _graph->insert(aten::log, {s[sp]});
        break;
      case cLog2:
        s[sp] = _graph->insert(aten::log2, {s[sp]});
        break;
      case cLog10:
        s[sp] = _graph->insert(aten::div, {_graph->insert(aten::log, {s[sp]}), const_log10});
        break;

      case cNeg:
        s[sp] = _graph->insert(aten::mul, {s[sp], const_minus_one});
        break;

      case cSqr:
        s[sp] = _graph->insert(aten::mul, {s[sp], s[sp]});
        break;
      case cSqrt:
        s[sp] = _graph->insert(aten::sqrt, {s[sp]});
        break;
      case cRSqrt:
        s[sp] = _graph->insert(aten::pow, {s[sp], const_minus_one_half});
        break;
      case cPow:
        --sp;
        s[sp] = _graph->insert(aten::pow, {s[sp], s[sp + 1]});
        break;
      case cExp:
        s[sp] = _graph->insert(aten::exp, {s[sp]});
        break;
      case cExp2:
        s[sp] = _graph->insert(aten::pow, {const_two, s[sp]});
        break;
      case cCbrt:
        s[sp] = _graph->insert(aten::pow, {s[sp], const_one_third});
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
          throw std::runtime_error("JIT Opcode not supported for libtorch tensors.");
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

  // build stack
  Stack stack;
  for (const auto & i : params)
    stack.push_back(i);

  if (_input.size() != params.size())
    throw std::runtime_error("Unexpected number of inputs in ParsedJITTensor::Eval.");

  // disable autograd
  torch::NoGradGuard no_grad;

  if (!_graph_executor)
    _graph_executor = std::make_shared<GraphExecutor>(_graph, "F");

  // if (!_execution_plan)
  //   _execution_plan = std::make_shared<ExecutionPlan>(_graph_executor->getPlanFor(stack, 10));

  // InterpreterState(_execution_plan->code).run(stack);
  _graph_executor->run(stack);

  // exec.run(stack);
  if (stack.size() != 1)
    throw std::runtime_error("Unexpected number vof outputs in ParsedJITTensor::Eval.");

  torch::autograd::GradMode::set_enabled(ad);

  return stack[0].toTensor();
}
