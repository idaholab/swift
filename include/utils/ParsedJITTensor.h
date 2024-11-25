/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

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
