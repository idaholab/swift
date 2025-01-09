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
 * Single relaxation time BGK collision for Lattice Boltzmann Method
 */
class LBMBGKCollision : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMBGKCollision(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  Real _tau_bgk;
  const torch::Tensor & _feq;
  const torch::Tensor & _f;
};
