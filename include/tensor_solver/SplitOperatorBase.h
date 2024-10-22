//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "TensorSolver.h"

/**
 * TensorTimeIntegrator object (this is mostly a compute object)
 */
class SplitOperatorBase : public TensorSolver
{
public:
  static InputParameters validParams();

  SplitOperatorBase(const InputParameters & parameters);

protected:
  const unsigned int _history_size;

  struct Variable
  {
    torch::Tensor & _buffer;
    const torch::Tensor & _reciprocal_buffer;
    const torch::Tensor & _linear_reciprocal;
    const torch::Tensor & _nonlinear_reciprocal;
    const std::vector<torch::Tensor> & _old_reciprocal_buffer;
    const std::vector<torch::Tensor> & _old_nonlinear_reciprocal;
  };

  std::vector<Variable> _variables;
};
