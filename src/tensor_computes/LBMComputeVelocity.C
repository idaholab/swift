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

using namespace torch::indexing;

registerMooseObject("SwiftApp", LBMComputeVelocity);

InputParameters
LBMComputeVelocity::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addRequiredParam<TensorInputBufferName>("f", "Distribution function");
  params.addRequiredParam<TensorInputBufferName>("rho", "Density");
  params.addParam<TensorInputBufferName>("forces", "forces", "Force tensor");
  params.addParam<std::string>("body_force", "0.0", "Body force to be added in x-dir");
  params.addClassDescription("Compute object for macroscopic velocity reconstruction.");
  return params;
}

LBMComputeVelocity::LBMComputeVelocity(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _f(getInputBuffer("f")),
    _rho(getInputBuffer("rho")),
    _force_tensor(getInputBuffer("forces")),
    _body_force_constant(_lb_problem.getConstant<Real>(getParam<std::string>("body_force")))
{
}

void
LBMComputeVelocity::computeBuffer()
{
  const unsigned int & dim = _domain.getDim();
  switch (dim)
  {
    case 3:
      _u.index_put_({Slice(), Slice(), Slice(), 0}, torch::sum(_f * _stencil._ex, 3) / _rho);
      _u.index_put_({Slice(), Slice(), Slice(), 1}, torch::sum(_f * _stencil._ey, 3) / _rho);
      _u.index_put_({Slice(), Slice(), Slice(), 2}, torch::sum(_f * _stencil._ez, 3) / _rho);
      break;
    case 2:
      _u.index_put_({Slice(), Slice(), Slice(), 0}, torch::sum(_f * _stencil._ex, 3) / _rho);
      _u.index_put_({Slice(), Slice(), Slice(), 1}, torch::sum(_f * _stencil._ey, 3) / _rho);
      break;
    default:
      mooseError("Unsupported dimension");
  }
  // include forces
  _u = _u + (_force_tensor + _body_force_constant) / (2.0 * _rho.unsqueeze(3));
  _lb_problem.maskedFillSolids(_u, 0);
}
