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
  params.addRequiredParam<TensorInputBufferName>("velocity", "LBM Velocty in x, y and z directions");
  return params;
}

LBMEquilibrium::LBMEquilibrium(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
  _rho(getInputBuffer("rho")),
  _velocity(getInputBuffer("velocity"))
{ 
}

void
LBMEquilibrium::computeBuffer()
{ 
  // prepping
  const unsigned int & dim = _mesh.getDim();
  torch::Tensor rho_unsqueezed = _rho.unsqueeze(3);
  torch::Tensor ux = _velocity.select(3, 0).unsqueeze(3);
  torch::Tensor uy = _velocity.select(3, 1).unsqueeze(3);
  torch::Tensor uz;

  switch (dim)
  {
    case 3:
      uz = _velocity.select(3, 2).unsqueeze(3);
      break;
    case 2:
      uz = torch::zeros_like(rho_unsqueezed,  MooseTensor::floatTensorOptions());
      break;
    default:
      mooseError("Unsupported dimensions for buffer _u");
  }
  
  // compute equilibrium
  _u = _w * rho_unsqueezed * (1.0 + (_ex * ux + _ey * uy + _ez * uz) / _lb_problem._cs2 \
                         + 0.5 * ((_ex * ux + _ey * uy + _ez * uz) * (_ex * ux + _ey * uy + _ez * uz)) / _lb_problem._cs4 \
                          -0.5 * (ux * ux + uy * uy + uz * uz) / _lb_problem._cs2);
  _lb_problem.maskedFillSolids(_u, 0);
}
