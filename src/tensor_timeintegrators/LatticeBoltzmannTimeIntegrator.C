/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LatticeBoltzmannTimeIntegrator.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

InputParameters
LatticeBoltzmannTimeIntegrator::validParams()
{
  InputParameters params = TensorTimeIntegrator::validParams();
  params.addClassDescription("LatticeBoltzmannTimeIntegrator object to handle streaming.");
  return params;
}

LatticeBoltzmannTimeIntegrator::LatticeBoltzmannTimeIntegrator(const InputParameters & parameters)
  : TensorTimeIntegrator(parameters),
  _lb_problem(dynamic_cast<LatticeBoltzmannProblem&>(_tensor_problem)),
  _stencil(_lb_problem.getStencil())
{
}
