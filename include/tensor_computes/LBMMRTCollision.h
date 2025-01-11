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
 * Multi relaxation time collision for Lattice Boltzmann Method
 */
class LBMMRTCollision : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMMRTCollision(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const torch::Tensor & _feq;
  const torch::Tensor & _f;
};
