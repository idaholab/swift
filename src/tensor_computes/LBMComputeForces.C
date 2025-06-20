/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMComputeForces.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("SwiftApp", LBMComputeForces);

InputParameters
LBMComputeForces::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  // params.addRequiredParam<TensorInputBufferName>("f", "Distribution function");
  params.addRequiredParam<TensorInputBufferName>("velocity", "Macroscopic velocity");
  params.addRequiredParam<TensorInputBufferName>("rho", "Macroscopic density");
  params.addParam<bool>("enable_gravity", false, "Whether to consider gravity or not");

  params.addClassDescription("Compute object for LB forces");
  return params;
}

LBMComputeForces::LBMComputeForces(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters), /*_f(getInputBuffer("f"))*/
    _velocity(getInputBuffer(getParam<TensorInputBufferName>("velocity"))),
    _density(getInputBuffer(getParam<TensorInputBufferName>("rho"))),
    _enable_gravity(getParam<bool>("enable_gravity")),
    _g(-9.81 * _lb_problem.getConstant<Real>("C_t") * _lb_problem.getConstant<Real>("C_t") /
       _lb_problem.getConstant<Real>("dx")),
    _tau(_lb_problem.getConstant<Real>("tau"))
{
  _source_term = torch::zeros_like(_u);
  _body_force = torch::zeros_like(_velocity);
}

void
LBMComputeForces::computeBodyForce()
{
  if (_enable_gravity)
    _body_force
        .index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 1})
        .fill_(-1.0 * _g * _density);
  else
  {
    // TBD
  }
}

void
LBMComputeForces::computeSourceTerm()
{
  const unsigned int & dim = _domain.getDim();

  if (_density.dim() < 3)
    _density.unsqueeze_(2);

  torch::Tensor rho_unsqueezed = _density.unsqueeze(3);
  torch::Tensor Fx = _body_force.select(3, 0).unsqueeze(3);
  torch::Tensor Fy = _body_force.select(3, 1).unsqueeze(3);
  torch::Tensor Fz;

  torch::Tensor ux = _velocity.select(3, 0).unsqueeze(3);
  torch::Tensor uy = _velocity.select(3, 1).unsqueeze(3);
  torch::Tensor uz;

  torch::Tensor e_xyz = torch::stack({_stencil._ex, _stencil._ey, _stencil._ez}, 0);

  switch (dim)
  {
    case 3:
    {
      Fz = _body_force.select(3, 2).unsqueeze(3);
      uz = _velocity.select(3, 2).unsqueeze(3);
      break;
    }
    case 2:
    {
      Fz = torch::zeros_like(rho_unsqueezed, MooseTensor::floatTensorOptions());
      uz = torch::zeros_like(rho_unsqueezed, MooseTensor::floatTensorOptions());
      break;
    }
    default:
      mooseError("Unsupported dimensions for buffer _u");
  }

  torch::Tensor Fxyz = torch::stack({Fx, Fy, Fz}, 3);
  torch::Tensor Uxyz = torch::stack({ux, uy, uz}, 3);
  torch::Tensor Fxyz_expanded = Fxyz.unsqueeze(-1);       // Shape: (Nx, Ny, Nz, 3, 1)
  torch::Tensor Uxyz_expanded = Uxyz.unsqueeze(-2);       // Shape: (Nx, Ny, Nz, 1, 3)
  torch::Tensor UF_outer = Fxyz_expanded * Uxyz_expanded; // Shape: (Nx, Ny, Nz, 3, 3)
  torch::Tensor UF_outer_flat = UF_outer.flatten(-2, -1); // Shape: (Nx, Ny, Nz, 9)

  for (int64_t ic = 0; ic < _stencil._q; ic++)
  {
    auto exyz_ic = e_xyz.index({Slice(), ic}).flatten(); // Shape (3)
    torch::Tensor ccr_flat =
        torch::outer(exyz_ic, exyz_ic) / _lb_problem._cs2 -
        torch::eye(3, MooseTensor::floatTensorOptions()).flatten(); // Shape (9)

    torch::Tensor multiplied = UF_outer_flat * ccr_flat; // Shape: (Nx, Ny, Nz, 9)

    // sum along the last dimension
    torch::Tensor UFccr = multiplied.sum(/*dim=*/-1);

    // compute source
    _source_term.index_put_(
        {Slice(), Slice(), Slice(), ic},
        _stencil._weights[ic] *
            ((_stencil._ex[ic] * Fx + _stencil._ey[ic] * Fy + _stencil._ez[ic] * Fz) /
                 _lb_problem._cs2 +
             UFccr / 2.0 / _lb_problem._cs4));
  }
}

void
LBMComputeForces::computeBuffer()
{
  _u = _u + (1.0 - 1.0 / (2.0 * _tau)) * _source_term;
  // _lb_problem.maskedFillSolids(_u, 0);
}
