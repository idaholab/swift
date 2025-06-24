/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LatticeBoltzmannOperator.h"

/**
 * Compute forces
 */
class LBMComputeForces : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMComputeForces(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const torch::Tensor & _temperature;
  const Real & _density;
  const Real & _T0;
  const bool _enable_gravity;
  const Real _g; // gravitational acceleration
};
