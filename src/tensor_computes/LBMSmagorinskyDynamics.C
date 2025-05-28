/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMSmagorinskyDynamics.h"
#include "LatticeBoltzmannProblem.h"

registerMooseObject("SwiftApp", LBMSmagorinskyDynamics);

InputParameters
LBMSmagorinskyDynamics::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Compute object for for Smagorinsky relaxation.");
  params.addRequiredParam<TensorInputBufferName>("feq", "Equilibrium distribution");
  params.addRequiredParam<TensorInputBufferName>("f", "Pre-collision distribution");
  return params;
}

LBMSmagorinskyDynamics::LBMSmagorinskyDynamics(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
  _feq(getInputBuffer("feq")),
  _f(getInputBuffer("f")),
  _shape(_lb_problem.getGridSize())
{
}

void
LBMSmagorinskyDynamics::computeBuffer()
{
  /// TBD
}
