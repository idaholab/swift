/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperatorBase.h"

/**
 * Apply stabilization to linear and nonlinear terms
 */
class ApplyStabilization : public TensorOperatorBase
{
public:
  static InputParameters validParams();

  ApplyStabilization(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const torch::Tensor & _linear;
  const torch::Tensor & _nonlinear;

  const torch::Tensor & _reciprocal;
  const torch::Tensor & _stabilization;

  torch::Tensor & _stabilized_linear;
  torch::Tensor & _stabilized_nonlinear;
};
