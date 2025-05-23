/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include <ATen/core/TensorBody.h>
#ifdef NEML2_ENABLED

#include "TensorOperator.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"

/**
 * Constant Tensor
 */
class FFTMechanics : public TensorOperator<>
{
public:
  static InputParameters validParams();

  FFTMechanics(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  // // stiffness
  // neml2::SSR4 _C;

  /// R2 identity tensor
  const torch::Tensor _ti;
  const torch::Tensor _tI;
  const torch::Tensor _tI4;
  const torch::Tensor _tI4rt;
  const torch::Tensor _tI4s;
  const torch::Tensor _tII;

  /// Material parameters
  const torch::Tensor & _tK;
  const torch::Tensor & _tmu;

  const std::vector<int64_t> _r2_shape;

  /// Gamma projection operator
  torch::Tensor _Ghat4;

  /// stress
  torch::Tensor & _tP;

  Real _l_tol;
  unsigned int _l_max_its;
  Real _nl_rel_tol;
  Real _nl_abs_tol;
  unsigned int _nl_max_its;

  using TensorOperatorBase::_dim;
  using TensorOperatorBase::_domain;
};

#endif
