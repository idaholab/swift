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
 * Swift-Hohenberg linear operator
 */
class SwiftHohenbergLinear : public TensorOperator
{
public:
  static InputParameters validParams();

  SwiftHohenbergLinear(const InputParameters & parameters);

  virtual void computeBuffer() override;

  const Real _r;
  const Real _alpha;
  const torch::Tensor & _k2;
};
