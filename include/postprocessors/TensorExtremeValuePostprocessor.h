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
 * Compute the average of an FFT buffer
 */
class TensorExtremeValuePostprocessor : public TensorPostprocessor
{
public:
  static InputParameters validParams();

  TensorExtremeValuePostprocessor(const InputParameters & parameters);

  virtual void initialize() override {}
  virtual void execute() override;
  virtual void finalize() override {}
  virtual PostprocessorValue getValue() const override;

protected:
  enum class ValueType
  {
    MIN,
    MAX
  } _value_type;

  Real _value;
};