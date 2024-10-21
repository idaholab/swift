//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "TensorPostprocessor.h"

/**
 * Compute the integral of a Tensor buffer
 */
class TensorIntegralChangePostprocessor : public TensorPostprocessor
{
public:
  static InputParameters validParams();

  TensorIntegralChangePostprocessor(const InputParameters & parameters);

  virtual void initialize() override {}
  virtual void execute() override;
  virtual void finalize() override {}
  virtual PostprocessorValue getValue() const override;

protected:
  const std::vector<torch::Tensor> & _u_old;
  Real _integral;
};
