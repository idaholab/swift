/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#ifdef NEML2_ENABLED

#include "TensorOperator.h"
#include "neml2/tensors/Vec.h"

/**
 * Hyperelastic isotropic material model
 */
class HyperElasticIsotropic : public TensorOperator<>
{
public:
  static InputParameters validParams();

  HyperElasticIsotropic(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  /// R2 identity tensor
  const torch::Tensor _ti;
  const torch::Tensor _tI;
  const torch::Tensor _tI4;
  const torch::Tensor _tI4rt;
  const torch::Tensor _tI4s;
  const torch::Tensor _tII;

  const torch::Tensor & _tF;
  const torch::Tensor & _tmu;
  const torch::Tensor & _tK;
  torch::Tensor & _tK4;
};

#endif
