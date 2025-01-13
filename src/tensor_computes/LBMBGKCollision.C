/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMBGKCollision.h"
#include "LatticeBoltzmannProblem.h"

registerMooseObject("SwiftApp", LBMBGKCollision);

InputParameters
LBMBGKCollision::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Compute Object for single relaxation BGK collision for Lattice Boltzmann Method.");
  params.addParam<Real>("tau", 1.0, "Relaxation parameter");
  params.addRequiredParam<TensorInputBufferName>("feq", "Equilibrium distribution");
  params.addRequiredParam<TensorInputBufferName>("f", "Pre-collision distribution");
  return params;
}

LBMBGKCollision::LBMBGKCollision(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
  _tau_bgk(getParam<Real>("tau")),
  _feq(getInputBuffer("feq")),
  _f(getInputBuffer("f"))
{
}

void
LBMBGKCollision::computeBuffer()
{
  _u = _f - 1.0 / _tau_bgk * (_f - _feq);
  _lb_problem.maskedFillSolids(_u, 0);
}
