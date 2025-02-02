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
 * Multi relaxation time collision with Hermite polynomials for Lattice Boltzmann Method
 */
class LBMRegularizedMRTCollision : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMRegularizedMRTCollision(const InputParameters & parameters);

  const torch::Tensor & regularize();

  void enableSlip();

  void computeBuffer() override;

protected:
  const torch::Tensor & _feq;
  const torch::Tensor & _f;
  const torch::IntArrayRef &_shape;
};
