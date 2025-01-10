/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMComputeVelocity.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"
#include "LatticeBoltzmannMesh.h"

using namespace torch::indexing;

registerMooseObject("SwiftApp", LBMComputeVelocity );

InputParameters
LBMComputeVelocity::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addRequiredParam<TensorInputBufferName>("f", "Distribution function");
  params.addRequiredParam<TensorInputBufferName>("rho", "Density");
  params.addParam<Real>("body_force", 0.0, "Body force to be added to x-dir");
  params.addClassDescription("Compute object for macroscopic velocity reconstruction.");
  return params;
}

LBMComputeVelocity::LBMComputeVelocity (const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
  _f(getInputBuffer("f")),
  _rho(getInputBuffer("rho")),
  _body_force(getParam<Real>("body_force"))
{
}

void
LBMComputeVelocity::computeBuffer()
{   
  const unsigned int & dim = _mesh.getDim();
  switch (dim)
  {
    case 3:
      _u.index_put_({Slice(), Slice(), Slice(), 0}, torch::sum(_f * _stencil._ex, 3) / _rho + _body_force) ;
      _u.index_put_({Slice(), Slice(), Slice(), 1}, torch::sum(_f * _stencil._ey, 3) / _rho);
      _u.index_put_({Slice(), Slice(), Slice(), 2}, torch::sum(_f * _stencil._ez, 3) / _rho);
      break;
    case 2:
      _u.index_put_({Slice(), Slice(), Slice(), 0}, torch::sum(_f * _stencil._ex, 3) / _rho + _body_force);
      _u.index_put_({Slice(), Slice(), Slice(), 1}, torch::sum(_f * _stencil._ey, 3) / _rho);
      break;
    default:
      mooseError("Unsupported dimension");
  }
  _lb_problem.setTensorToValue(_u, 0);
}
