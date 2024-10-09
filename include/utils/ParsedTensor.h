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
#include "torch/torch.h"

class ParsedTensor : public FunctionParserAD
{
public:
  ParsedTensor();

  void setupTensors();

  /// overload for torch tensors
  torch::Tensor Eval(const std::vector<const torch::Tensor *> & params);

protected:
  /// dummy function for fourier transforms (those are executed in
  /// the custom bytecode interpreter instead)
  static Real fp_dummy(const Real *);

  /// we'll need a stack pool to make this thread safe
  std::vector<torch::Tensor> s;

  /// immediate values converted to tensors
  std::vector<torch::Tensor> tensor_immed;

  const Data & _data;

  std::size_t _mFFT;
  std::size_t _miFFT;
};
