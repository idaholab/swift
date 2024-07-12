//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "TensorOperatorBase.h"

/**
 * TensorOperator object with a single output
 */
class TensorOperator : public TensorOperatorBase
{
public:
  static InputParameters validParams();

  TensorOperator(const InputParameters & parameters);

protected:
  /// output buffer
  torch::Tensor & _u;
};
