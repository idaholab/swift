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
 * Compute object for macroscopic velocity reconstruction
 */
class LBMComputeVelocity : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMComputeVelocity(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const torch::Tensor & _f;
  const torch::Tensor & _rho;
  const torch::Tensor & _force_tensor;
  const Real & _body_force_constant;
  bool _is_force_applied;
};
