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
#include "FunctionInterface.h"

class Function;

/**
 * Constant Tensor
 */
class MooseFunctionTensor : public TensorOperator, public FunctionInterface
{
public:
  static InputParameters validParams();

  MooseFunctionTensor(const InputParameters & parameters);

  virtual void computeBuffer() override;

  const Function & _func;
};
