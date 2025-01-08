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
  _rho(getInputBuffer("rho")),
  _velocity(getInputBuffer("velocty")),
  _dim(_mesh.getDim())
{
  _shape = {1, 1, 1, _stencil._q};
}

void
LBMEquilibrium::computeBuffer()
{
  // preparing
  const torch::Tensor ex = _stencil._ex.view(_shape);
  const torch::Tensor ey = _stencil._ey.view(_shape);
  const torch::Tensor ez = _stencil._ez.view(_shape);
  const torch::Tensor w = _stencil._weights.view(_shape);
  torch::Tensor uz;
  switch (_dim)
  {
    case 3:
      uz = _velocity.select(3, 2).unsqueeze(3);
      break;
    case 2:
      uz = torch::zeros_like(_rho,  MooseTensor::floatTensorOptions()).unsqueeze(3);
      break;
    default:
      mooseError("Unsupported dimensions for buffer _u");
  }

  const torch::Tensor ux = _velocity.select(3, 0).unsqueeze(3);
  const torch::Tensor uy = _velocity.select(3, 1).unsqueeze(3);
  const torch::Tensor rho = _rho.unsqueeze(3);

  //
  _u = w * rho * (1.0 + (ex * ux + ey * uy + ez * uz) / _lb_problem._cs2 \
                         + 0.5 * ((ex * ux + ey * uy + ez * uz) * (ex * ux + ey * uy + ez * uz)) / _lb_problem._cs4 \
                          -0.5 * (ux * ux + uy * uy + uz * uz) / _lb_problem._cs2);
  _lb_problem.setTensorToValue(_u, 0);
}

