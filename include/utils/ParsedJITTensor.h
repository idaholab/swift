//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "libmesh/fparser_ad.hh"
#include <torch/csrc/jit/ir/ir.h>
#include "torch/torch.h"

namespace torch
{
namespace jit
{
struct Graph;
struct GraphExecutor;
struct Value;
struct ExecutionPlan;
}
}

class ParsedJITTensor : public FunctionParserAD
{
public:
  ParsedJITTensor();

  void setupTensors();

  /// overload for torch tensors
  torch::Tensor Eval(const std::vector<const torch::Tensor *> & params);

  /// print IR for debugging
  void print() { _graph->dump(); }

protected:
  /// graph input nodes
  std::vector<torch::jit::Value *> _input;

  /// immediate values converted to tensors
  std::vector<torch::jit::Value *> _constant_immed;

  /// compute graph
  std::shared_ptr<torch::jit::Graph> _graph;
  std::shared_ptr<torch::jit::GraphExecutor> _graph_executor;
  std::shared_ptr<torch::jit::ExecutionPlan> _execution_plan;

  const Data & _data;
};
