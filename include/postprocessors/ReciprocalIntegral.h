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
 * Fetch teh zero k-vector component
 */
class ReciprocalIntegral : public TensorPostprocessor
{
public:
  static InputParameters validParams();

  ReciprocalIntegral(const InputParameters & parameters);

  virtual void initialize() override {}
  virtual void execute() override;
  virtual void finalize() override;
  virtual PostprocessorValue getValue() const override;

protected:
  Real _integral;
};
