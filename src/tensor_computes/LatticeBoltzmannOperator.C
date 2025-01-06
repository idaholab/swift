/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LatticeBoltzmannOperator.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"
#include "LatticeBoltzmannMesh.h"

InputParameters
LatticeBoltzmannOperator::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription("LatticeBoltzmannOperator object.");
  return params;
}

LatticeBoltzmannOperator::LatticeBoltzmannOperator(const InputParameters & parameters)
  : TensorOperator(parameters),
  _lb_problem(dynamic_cast<LatticeBoltzmannProblem&>(_tensor_problem)),
  _stencil(_lb_problem.getStencil()),
  _mesh(dynamic_cast<LatticeBoltzmannMesh &>(_lb_problem.mesh()))
{
}

