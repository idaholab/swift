//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "FFTTimeIntegrator.h"

/**
 * FFTTimeIntegrator object (this is mostly a compute object)
 */
class FFTSemiImplicit : public FFTTimeIntegrator
{
public:
  static InputParameters validParams();

  FFTSemiImplicit(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const unsigned int _history_size;
  const torch::Tensor & _reciprocal_buffer;
  const torch::Tensor & _linear_reciprocal;
  const torch::Tensor & _non_linear_reciprocal;
  const std::vector<torch::Tensor> & _old_reciprocal_buffer;
  const std::vector<torch::Tensor> & _old_non_linear_reciprocal;
};
