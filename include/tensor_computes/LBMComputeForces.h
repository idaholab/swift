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

  void computeSourceTerm();
  void computeBodyForce();
  void computeBuffer() override;

protected:
  const torch::Tensor & _velocity;
  const torch::Tensor & _density;

  torch::Tensor _source_term;
  torch::Tensor _body_force;

  const bool _enable_gravity;
  const Real _g; // gravitational acceleration
  const Real _tau;
  // const torch::Tensor & _f;
};
