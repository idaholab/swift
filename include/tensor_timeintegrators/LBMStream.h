/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LatticeBoltzmannTimeIntegrator.h"
#include "LatticeBoltzmannStencilBase.h"

/**
 * LBM Stream object
 */
class LBMStream : public LatticeBoltzmannTimeIntegrator
{
public:
  static InputParameters validParams();

  LBMStream(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const std::vector<torch::Tensor> &  _f_old;
};
