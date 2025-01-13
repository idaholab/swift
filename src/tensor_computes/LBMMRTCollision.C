/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMMRTCollision.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

registerMooseObject("SwiftApp", LBMMRTCollision);

InputParameters
LBMMRTCollision::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Compute Object for multi relaxation time collision for Lattice Boltzmann Method.");
  params.addRequiredParam<TensorInputBufferName>("feq", "Equilibrium distribution");
  params.addRequiredParam<TensorInputBufferName>("f", "Pre-collision distribution");
  return params;
}

LBMMRTCollision::LBMMRTCollision(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
  _feq(getInputBuffer("feq")),
  _f(getInputBuffer("f"))
{
}

void
LBMMRTCollision::computeBuffer()
{
  //
  const auto shape = _u.sizes();
  // f = M^-1 x S x M x (f - feq)
  _u = _f - torch::matmul(_stencil._M_inv,
                            torch::matmul(_stencil._S,
                            torch::matmul(_stencil._M,
                            (_f - _feq).view({-1, _stencil._q}).t()))).t().view({shape});
  _lb_problem.maskedFillSolids(_u, 0);
}
