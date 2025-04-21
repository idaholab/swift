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
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"

/**
 * Constant Tensor
 */
class MoulinecSouquet : public TensorOperator<neml2::SR2>
{
public:
  static InputParameters validParams();

  MoulinecSouquet(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  void updateGamma();

  /// stiffness estimate
  neml2::SSR4 _C0;

  /// Gamma operator
  torch::Tensor _gamma;

  /// constants used for building Gamma
  const std::array<const torch::Tensor *, 3> _kvec;
  const std::array<std::pair<int, int>, 6> _map;
  const std::vector<Real> _f;
  const std::vector<Real> _inv_f;
};

#endif
