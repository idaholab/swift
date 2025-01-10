/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorTimeIntegrator.h"

class LatticeBoltzmannProblem;
class LatticeBoltzmannStencilBase;

/**
 * TimeIntegrator object for LB computes to handle streaming operation
 */
class LatticeBoltzmannTimeIntegrator : public TensorTimeIntegrator
{
public:
  static InputParameters validParams();

  LatticeBoltzmannTimeIntegrator(const InputParameters & parameters);

protected:
  LatticeBoltzmannProblem& _lb_problem;
  const LatticeBoltzmannStencilBase & _stencil;
};
