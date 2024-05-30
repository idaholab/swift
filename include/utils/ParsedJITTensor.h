//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "NEML2Utils.h"
#include "neml2/tensors/Scalar.h"

#include "libmesh/fparser_ad.hh"

namespace torch
{
namespace jit
{
struct Graph;
struct Value;
}
}

class ParsedJITTensor : public FunctionParserAD
{
public:
  ParsedJITTensor();

  void setupTensors();

  /// overload for torch tensors
  neml2::Scalar Eval(const neml2::Scalar * params);

protected:
  /// immediate values converted to tensors
  std::vector<torch::jit::Value *> _input;

  /// immediate values converted to tensors
  std::vector<torch::jit::Value *> _constant_immed;

  /// compute graph
  std::shared_ptr<torch::jit::Graph> _graph;

  const Data & _data;
};

/*
https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/codegen/fuser/interface.h
https://github.com/pytorch/pytorch/blob/main/test/cpp/jit/test_fuser.cpp#L272
https://github.com/pytorch/pytorch/blob/main/test/cpp/jit/test_graph_executor.cpp
*/
