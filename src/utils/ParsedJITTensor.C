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
#include "Conversion.h"
#include "SwiftUtils.h"
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

namespace
{
const std::string
FP_GetOpcodeName(int opcode)
{
  using namespace FUNCTIONPARSERTYPES;

  /* Symbolic meanings for the opcodes? */
  const char * p = 0;
  switch (opcode)
  {
    case cAbs:
      p = "cAbs";
      break;
    case cAcos:
      p = "cAcos";
      break;
    case cAcosh:
      p = "cAcosh";
      break;
    case cArg:
      p = "cArg";
      break;
    case cAsin:
      p = "cAsin";
      break;
    case cAsinh:
      p = "cAsinh";
      break;
    case cAtan:
      p = "cAtan";
      break;
    case cAtan2:
      p = "cAtan2";
      break;
    case cAtanh:
      p = "cAtanh";
      break;
    case cCbrt:
      p = "cCbrt";
      break;
    case cCeil:
      p = "cCeil";
      break;
    case cConj:
      p = "cConj";
      break;
    case cCos:
      p = "cCos";
      break;
    case cCosh:
      p = "cCosh";
      break;
    case cCot:
      p = "cCot";
      break;
    case cCsc:
      p = "cCsc";
      break;
    case cExp:
      p = "cExp";
      break;
    case cExp2:
      p = "cExp2";
      break;
    case cFloor:
      p = "cFloor";
      break;
    case cHypot:
      p = "cHypot";
      break;
    case cIf:
      p = "cIf";
      break;
    case cImag:
      p = "cImag";
      break;
    case cInt:
      p = "cInt";
      break;
    case cLog:
      p = "cLog";
      break;
    case cLog2:
      p = "cLog2";
      break;
    case cLog10:
      p = "cLog10";
      break;
    case cMax:
      p = "cMax";
      break;
    case cMin:
      p = "cMin";
      break;
    case cPolar:
      p = "cPolar";
      break;
    case cPow:
      p = "cPow";
      break;
    case cReal:
      p = "cReal";
      break;
    case cSec:
      p = "cSec";
      break;
    case cSin:
      p = "cSin";
      break;
    case cSinh:
      p = "cSinh";
      break;
    case cSqrt:
      p = "cSqrt";
      break;
    case cTan:
      p = "cTan";
      break;
    case cTanh:
      p = "cTanh";
      break;
    case cTrunc:
      p = "cTrunc";
      break;
    case cImmed:
      p = "cImmed";
      break;
    case cJump:
      p = "cJump";
      break;
    case cNeg:
      p = "cNeg";
      break;
    case cAdd:
      p = "cAdd";
      break;
    case cSub:
      p = "cSub";
      break;
    case cMul:
      p = "cMul";
      break;
    case cDiv:
      p = "cDiv";
      break;
    case cMod:
      p = "cMod";
      break;
    case cEqual:
      p = "cEqual";
      break;
    case cNEqual:
      p = "cNEqual";
      break;
    case cLess:
      p = "cLess";
      break;
    case cLessOrEq:
      p = "cLessOrEq";
      break;
    case cGreater:
      p = "cGreater";
      break;
    case cGreaterOrEq:
      p = "cGreaterOrEq";
      break;
    case cNot:
      p = "cNot";
      break;
    case cAnd:
      p = "cAnd";
      break;
    case cOr:
      p = "cOr";
      break;
    case cDeg:
      p = "cDeg";
      break;
    case cRad:
      p = "cRad";
      break;
    case cFCall:
      p = "cFCall";
      break;
    case cPCall:
      p = "cPCall";
      break;
#ifdef FP_SUPPORT_OPTIMIZER
    case cFetch:
      p = "cFetch";
      break;
    case cPopNMov:
      p = "cPopNMov";
      break;
    case cLog2by:
      p = "cLog2by";
      break;
    case cNop:
      p = "cNop";
      break;
#endif
    case cSinCos:
      p = "cSinCos";
      break;
    case cSinhCosh:
      p = "cSinhCosh";
      break;
    case cAbsNot:
      p = "cAbsNot";
      break;
    case cAbsNotNot:
      p = "cAbsNotNot";
      break;
    case cAbsAnd:
      p = "cAbsAnd";
      break;
    case cAbsOr:
      p = "cAbsOr";
      break;
    case cAbsIf:
      p = "cAbsIf";
      break;
    case cDup:
      p = "cDup";
      break;
    case cInv:
      p = "cInv";
      break;
    case cSqr:
      p = "cSqr";
      break;
    case cRDiv:
      p = "cRDiv";
      break;
    case cRSub:
      p = "cRSub";
      break;
    case cNotNot:
      p = "cNotNot";
      break;
    case cRSqrt:
      p = "cRSqrt";
      break;
    case VarBegin:
      p = "VarBegin";
      break;
    default:
      throw std::runtime_error("Unknown opcode.");
  }
  std::ostringstream tmp;
  // if(!p) std::cerr << "o=" << opcode << "\n";
  assert(p);
  tmp << p;
  return tmp.str();
}
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
  const auto const_one_third = _graph->insertConstant(1.0 / 3.0);

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
        s[sp] = _graph->insert(aten::tan, {s[sp]});
        break;

      case cTanh:
        s[sp] = _graph->insert(aten::tanh, {s[sp]});
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
        s[sp] = _graph->insert(aten::log10, {s[sp]});
        break;

      case cNeg:
        s[sp] = _graph->insert(aten::neg, {s[sp]});
        break;

      case cSqr:
        s[sp] = _graph->insert(aten::mul, {s[sp], s[sp]});
        break;
      case cSqrt:
        s[sp] = _graph->insert(aten::sqrt, {s[sp]});
        break;
      case cRSqrt:
        s[sp] = _graph->insert(aten::rsqrt, {s[sp]});
        break;
      case cPow:
        --sp;
        s[sp] = _graph->insert(aten::pow, {s[sp], s[sp + 1]});
        break;
      case cExp:
        s[sp] = _graph->insert(aten::exp, {s[sp]});
        break;
      case cExp2:
        s[sp] = _graph->insert(aten::exp2, {s[sp]});
        break;
      case cCbrt:
        s[sp] = _graph->insert(aten::pow, {s[sp], const_one_third});
        break;

      case cFetch:
        ++sp;
        s[sp] = s[ByteCode[++i]];
        break;
      case cDup:
        ++sp;
        s[sp] = s[sp - 1];
        break;

#ifdef FP_SUPPORT_OPTIMIZER
      case cPopNMov:
      {
        int dst = ByteCode[++i], src = ByteCode[++i];
        s[dst] = s[src];
        sp = dst;
        break;
      }
      case cLog2by:
        --sp;
        s[sp] = _graph->insert(aten::mul, {_graph->insert(aten::log2, {s[sp]}), s[sp + 1]});
        break;
      case cNop:
        break;
#endif

      default:
        if (op >= VarBegin)
        {
          // // load variable
          ++sp;
          s[sp] = _input[op - VarBegin];
        }
        else
        {
          throw std::runtime_error("JIT Opcode " + FP_GetOpcodeName(op) +
                                   " not supported for libtorch tensors.");
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

torch::Tensor
ParsedJITTensor::Eval(const std::vector<const torch::Tensor *> & params)
{
  using namespace torch::jit;

  // build stack
  Stack stack;
  for (const auto & p : params)
    stack.push_back(*p);

  if (_input.size() != params.size())
    throw std::runtime_error("Unexpected number of inputs in ParsedJITTensor::Eval.");

  // disable autograd
  torch::NoGradGuard no_grad;

  if (!_graph_executor)
    _graph_executor = std::make_shared<GraphExecutor>(_graph, "F");

  _graph_executor->run(stack);

  if (stack.size() != 1)
    throw std::runtime_error("Unexpected number vof outputs in ParsedJITTensor::Eval.");

  return stack[0].toTensor();
}
