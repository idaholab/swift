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

/**
 * PerformFFT object
 */
template <bool forward>
class PerformFFTTempl : public TensorOperator
{
public:
  static InputParameters validParams();

  PerformFFTTempl(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const torch::Tensor & _input;

  using TensorOperator::_tensor_problem;
  using TensorOperator::_u;
};

typedef PerformFFTTempl<true> ForwardFFT;
typedef PerformFFTTempl<false> InverseFFT;
