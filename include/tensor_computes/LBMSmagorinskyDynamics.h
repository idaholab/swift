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
 * Compute object for for Smagorinsky relaxation
 */

class LBMSmagorinskyDynamics : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMSmagorinskyDynamics(const InputParameters & parameters);

  const torch::Tensor regularize();

  void enableSlip();

  void computeBuffer() override;

protected:
  const torch::Tensor & _feq;
  const torch::Tensor & _f;
  const std::array<int64_t, 3> _shape;
};
