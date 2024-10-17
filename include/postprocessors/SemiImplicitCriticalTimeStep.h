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
 * Compute the average of a Tensor buffer
 */
class SemiImplicitCriticalTimeStep : public TensorPostprocessor
{
public:
  static InputParameters validParams();

  SemiImplicitCriticalTimeStep(const InputParameters & parameters);

  virtual void initialize() override {}
  virtual void execute() override;
  virtual void finalize() override;
  virtual PostprocessorValue getValue() const override;

protected:
  Real _critical_dt;
};
