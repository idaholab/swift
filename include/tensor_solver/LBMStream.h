/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorSolver.h"

class LatticeBoltzmannProblem;
class LatticeBoltzmannStencilBase;

/**
 * LBM Stream object
 */
class LBMStream : public TensorSolver
{
public:
  static InputParameters validParams();

  LBMStream(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  LatticeBoltzmannProblem & _lb_problem;
  const LatticeBoltzmannStencilBase & _stencil;

  torch::Tensor & _u;
  const std::vector<torch::Tensor> & _f_old;
};
