/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMEquilibrium.h"

registerMooseObject("SwiftApp", LBMEquilibrium);

InputParameters
LBMEquilibrium::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Compute LB equilibrium distribution object.");
  params.addRequiredParam<TensorInputBufferName>("rho", "LBM Density");
  params.addRequiredParam<TensorInputBufferName>("velocty", "LBM Velocty in x, y and z directions");
  return params;
}

LBMEquilibrium::LBMEquilibrium(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
  _rho(getInputBuffer("rho").unsqueeze(3)),
  _velocity(getInputBuffer("velocty")),
  _ux(_velocity.select(3, 0).unsqueeze(3)),
  _uy(_velocity.select(3, 1).unsqueeze(3)),
  _dim(_mesh.getDim())
{
  // recunstruct uz
  switch (_dim)
  {
    case 3:
      _uz = _velocity.select(3, 2).unsqueeze(3);
      break;
    case 2:
      _uz = torch::zeros_like(_rho,  MooseTensor::floatTensorOptions());
      break;
    default:
      mooseError("Unsupported dimensions for buffer _u");
  }
}

void
LBMEquilibrium::computeBuffer()
{
  // compute equilibrium
  _u = _w * _rho * (1.0 + (_ex * _ux + _ey * _uy + _ez * _uz) / _lb_problem._cs2 \
                         + 0.5 * ((_ex * _ux + _ey * _uy + _ez * _uz) * (_ex * _ux + _ey * _uy + _ez * _uz)) / _lb_problem._cs4 \
                          -0.5 * (_ux * _ux + _uy * _uy + _uz * _uz) / _lb_problem._cs2);
  _lb_problem.setTensorToValue(_u, 0);
}
