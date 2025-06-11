/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorPostprocessor.h"

/**
 * Compute Reynolds number
 */
class ComputeReynoldsNumber : public TensorPostprocessor
{
public:
  static InputParameters validParams();

  ComputeReynoldsNumber(const InputParameters & parameters);

  virtual void initialize() override {}
  virtual void execute() override;
  virtual void finalize() override {}
  virtual PostprocessorValue getValue() const override;

protected:
  const Real _kinematic_viscosity;
  const Real _D; // diameter
  Real _Reynolds_number;
};
