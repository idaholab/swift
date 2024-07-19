//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "TensorOperator.h"
#include "ParsedTensor.h"
#include "ParsedJITTensor.h"

/**
 * ParsedCompute object
 */
class ParsedCompute : public TensorOperator
{
public:
  static InputParameters validParams();

  ParsedCompute(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const bool _use_jit;
  const bool _extra_symbols;
  std::vector<torch::Tensor> _constant_tensors;

  ParsedJITTensor _jit;
  ParsedTensor _no_jit;

  std::vector<const torch::Tensor *> _params;
};
