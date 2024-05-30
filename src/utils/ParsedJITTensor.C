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

  // make sure graph is well formed
  _graph->lint();

  _graph->dump();
}

neml2::Scalar
ParsedJITTensor::Eval(const neml2::Scalar * params)
{
  return params[0];
}
