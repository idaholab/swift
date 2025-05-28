/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMRegularize.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

registerMooseObject("SwiftApp", LBMRegularize);

InputParameters
LBMRegularize::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addClassDescription("Compute object to project non-equilibrium onto Hermite space.");
  params.addRequiredParam<TensorInputBufferName>("feq", "Equilibrium distribution");
  params.addRequiredParam<TensorInputBufferName>("f", "Pre-collision distribution");
  return params;
}

LBMRegularize::LBMRegularize(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
  _feq(getInputBuffer("feq")),
  _f(getInputBuffer("f")),
  _shape(_lb_problem.getGridSize())
{
  int64_t nx = _shape[0];
  int64_t ny = _shape[1];
  int64_t nz = _shape[2];

  // regularized non-equilibrium
  _f_neq_hat = torch::zeros({nx*ny*nz, _stencil._q}, MooseTensor::floatTensorOptions());
  
  // intermediate tensors
  _fneqtimescc = torch::zeros({nx * ny * nz, 9}, MooseTensor::floatTensorOptions());
  _e_xyz = torch::stack({_stencil._ex, _stencil._ey, _stencil._ez}, 0);

}

void
LBMRegularize::computeBuffer()
{
  /**
   * Regularization procedure projects non-equilibrium (fneq) distribution
   * onto the second order Hermite space.
   * For more information: https://doi.org/10.3390/fluids8010001
   */

  using torch::indexing::Slice;
  int64_t nx = _shape[0];
  int64_t ny = _shape[1];
  int64_t nz = _shape[2];

  // Flatten tensors for easier manipulation
  auto f_flat = _f.view({nx * ny * nz, _stencil._q});
  auto feq_flat = _feq.view({nx * ny * nz, _stencil._q});

  // Intermediate tensors
  torch::Tensor fneq = f_flat - feq_flat;

  // Compute tensor products
  for (int ic = 0; ic < _stencil._q; ic++)
  {
    auto exyz_ic = _e_xyz.index({Slice(), ic}).flatten();
    torch::Tensor ccr = torch::outer(exyz_ic, exyz_ic).flatten();
    _fneqtimescc += (fneq.select(1, ic).view({nx * ny * nz, 1}) * ccr.view({1, 9}));
  }

  // Compute Hermite tensor
  torch::Tensor H2 = torch::zeros({1, 9}, MooseTensor::floatTensorOptions());
  for (int ic = 0; ic < _stencil._q; ic++)
  {
    auto exyz_ic = _e_xyz.index({Slice(), ic}).flatten();
    torch::Tensor ccr = torch::outer(exyz_ic, exyz_ic) /
                         _lb_problem._cs2 - torch::eye(3, MooseTensor::floatTensorOptions());
    H2 = ccr.flatten().unsqueeze(0).expand({nx * ny * nz, 9});

    // Compute regularized non-equilibrium distribution
    _f_neq_hat.index_put_({Slice(), ic}, (_stencil._weights[ic] * (1.0 / (2.0 * _lb_problem._cs2)) *
                                 (_fneqtimescc * H2).sum(1)));
  }
  
  _u = _f_neq_hat.view({nx, ny, nz, _stencil._q});
}