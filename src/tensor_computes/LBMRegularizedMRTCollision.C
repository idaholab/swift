/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMRegularizedMRTCollision.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

registerMooseObject("SwiftApp", LBMRegularizedMRTCollision);

InputParameters
LBMRegularizedMRTCollision::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Compute Object for multi relaxation time collision with Hermite polynomials for Lattice Boltzmann Method.");
  params.addRequiredParam<TensorInputBufferName>("feq", "Equilibrium distribution");
  params.addRequiredParam<TensorInputBufferName>("f", "Pre-collision distribution");
  return params;
}

LBMRegularizedMRTCollision::LBMRegularizedMRTCollision(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
  _feq(getInputBuffer("feq")),
  _f(getInputBuffer("f")),
  _shape(_lb_problem.getGridSize())
{
}

const torch::Tensor &
LBMRegularizedMRTCollision::regularize()
{
  /**
   * Regularization procedure projects non-equilibrium (fneq) distribution
   * onto the second order Hermite space.
   * For more information: https://doi.org/10.3390/fluids8010001
   */

  using torch::indexing::Slice;
  // Get tensor dimensions
  int64_t nx = _shape[0];
  int64_t ny = _shape[1];
  int64_t nz = _shape[2];

  // Flatten tensors for easier manipulation
  auto f_flat = _f.view({nx * ny * nz, _stencil._q});
  auto feq_flat = _feq.view({nx * ny * nz, _stencil._q});
  torch::Tensor f_neq_hat = torch::zeros_like(f_flat, MooseTensor::floatTensorOptions());

  // Intermediate tensors
  torch::Tensor fneq = f_flat - feq_flat;
  torch::Tensor fneqtimescc = torch::zeros({nx * ny * nz, 9}, MooseTensor::floatTensorOptions());
  torch::Tensor e_xyz = torch::stack({_stencil._ex, _stencil._ey, _stencil._ez}, 0);

  // Compute tensor products
  for (int ic = 0; ic < _stencil._q; ic++)
  {
    auto exyz_ic = e_xyz.index({Slice(), ic}).unsqueeze(0);
    torch::Tensor ccr = torch::bmm(exyz_ic.unsqueeze(2), exyz_ic.unsqueeze(1)).squeeze().flatten();
    fneqtimescc += (fneq.select(1, ic).view({nx * ny * nz, 1}) * ccr.view({1, 9}));
  }

  // Compute Hermite tensor
  torch::Tensor H2 = torch::zeros({1, 9}, MooseTensor::floatTensorOptions());
  for (int ic = 0; ic < _stencil._q; ic++)
  {
    auto exyz_ic = e_xyz.index({Slice(), ic}).unsqueeze(0);
    torch::Tensor ccr = torch::bmm(exyz_ic.unsqueeze(2), exyz_ic.unsqueeze(1)).squeeze() /
                         _lb_problem._cs2 - torch::eye(3);
    H2 = ccr.flatten().unsqueeze(0).expand({nx * ny * nz, 9});

    // Compute regularized non-equilibrium distribution
    f_neq_hat.index_put_({Slice(), ic}, (_stencil._weights[ic] * (1.0 / (2.0 * _lb_problem._cs2)) *
                                 (fneqtimescc * H2).sum(1)));
  }

  // Back to the correct shape
  return f_neq_hat.view({nx, ny, nz, _stencil._q});
}

void
LBMRegularizedMRTCollision::enableSlip()
{
  /**
   * Regularized MRT collision with slip relaxation matrix
   */
  const torch::Tensor & relaxation_matrix = _lb_problem.getSlipRelaxationMatrix();
  torch::Tensor f_neq_hat = regularize();

  // Reshape
  torch::Tensor f_neq_reshaped = f_neq_hat.view({-1, _stencil._q, 1});  // Shape: (Nx*Ny*Nz, q, 1)
  torch::Tensor S_reshaped = relaxation_matrix.view({-1, _stencil._q, _stencil._q});  // Shape: (Nx*Ny*Nz, q, q)

  // Transform f_neq_hat to moment space and apply relaxation
  torch::Tensor temp = torch::bmm(_stencil._M.expand({S_reshaped.size(0), _stencil._q, _stencil._q}), f_neq_reshaped);  // Shape: (Nx*Ny*Nz, q, 1)
  temp = torch::bmm(S_reshaped, temp);  // Shape: (Nx*Ny*Nz, q, 1)
  temp = torch::bmm(_stencil._M_inv.expand({S_reshaped.size(0), _stencil._q, _stencil._q}), temp);  // Shape: (Nx*Ny*Nz, q, 1)

  // Reshape temp to match the original shape
  torch::Tensor update = temp.view(f_neq_hat.sizes());  // Shape: (Nx, Ny, Nz, q)

  // Final update calculation
  _u = _feq + f_neq_hat - update;
}

void
LBMRegularizedMRTCollision::computeBuffer()
{

  const bool & is_slip_enabled = _lb_problem.isSlipEnabled();

  if (is_slip_enabled)
    enableSlip();

  else
  {
    const torch::Tensor & f_neq_hat = regularize();
    _u = _feq + f_neq_hat - torch::matmul(_stencil._M_inv,
                              torch::matmul(_stencil._S,
                              torch::matmul(_stencil._M,
                              f_neq_hat.view({-1, _stencil._q}).t()))).t().view({f_neq_hat.sizes()});
  }

  _lb_problem.maskedFillSolids(_u, 0);
}
