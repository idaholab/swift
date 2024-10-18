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
 * Constant Tensor
 */
class FFTGradientSquare : public TensorOperator
{
public:
  static InputParameters validParams();

  FFTGradientSquare(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _input;
  const bool _input_is_reciprocal;
  const Real _factor;

  /// mesh dimension
  const unsigned int & _dim;

  /// imaginary unit i
  const torch::Tensor _i;
};
