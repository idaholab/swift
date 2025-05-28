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
  * Compute object to project non-equilibrium onto Hermite space
  */
class LBMRegularize : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMRegularize(const InputParameters & parameters);

  const torch::Tensor regularize();

  void enableSlip();

  void computeBuffer() override;

protected:
  const torch::Tensor & _feq;
  const torch::Tensor & _f;
  const std::array<int64_t, 3> _shape;

  torch::Tensor _f_neq_hat;
  torch::Tensor _fneqtimescc;
  torch::Tensor _e_xyz;
};