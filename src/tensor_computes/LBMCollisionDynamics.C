/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMCollisionDynamics.h"

registerMooseObject("SwiftApp", LBMBGKCollision);
registerMooseObject("SwiftApp", LBMMRTCollision);
registerMooseObject("SwiftApp", LBMSmagorinskyCollision);

template <int coll_dyn>
InputParameters
LBMCollisionDynamicsTempl<coll_dyn>::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();

  params.addClassDescription("Template object for LBM collision dynamics");
  params.addRequiredParam<TensorInputBufferName>("f", "Input buffer distribution function");
  params.addRequiredParam<TensorInputBufferName>("feq",
                                                 "Input buffer equilibrium distribution function");
  params.addParam<bool>(
      "projection", false, "Whether or not to project non-equilibrium onto Hermite space.");

  return params;
}

template <int coll_dyn>
LBMCollisionDynamicsTempl<coll_dyn>::LBMCollisionDynamicsTempl(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
    _f(getInputBuffer("f")),
    _feq(getInputBuffer("feq")),
    _shape(_lb_problem.getGridSize()),
    _tau_0(_lb_problem.getConstant<Real>("tau")),
    _C_s(_lb_problem.getConstant<Real>("Cs")),
    _delta_x(1.0), //_lb_problem.getConstant("dx")
    _projection(getParam<bool>("projection"))
{
  //
  _fneq = torch::zeros({_shape[0], _shape[1], _shape[2], _stencil._q},
                       MooseTensor::floatTensorOptions());
}

template <int coll_dyn>
void
LBMCollisionDynamicsTempl<coll_dyn>::HermiteRegularization()
{
  /**
   * Regularization procedure projects non-equilibrium (fneq) distribution
   * onto the second order Hermite space.
   */

  using torch::indexing::Slice;

  int64_t nx = _shape[0];
  int64_t ny = _shape[1];
  int64_t nz = _shape[2];

  auto f_flat = _f.view({nx * ny * nz, _stencil._q});
  auto feq_flat = _feq.view({nx * ny * nz, _stencil._q});
  auto f_neq_hat = _fneq.view({nx * ny * nz, _stencil._q});

  torch::Tensor fneq = f_flat - feq_flat;
  torch::Tensor fneqtimescc =
      torch::zeros({nx * ny * nz, _stencil._q}, MooseTensor::floatTensorOptions());
  torch::Tensor e_xyz = torch::stack({_stencil._ex, _stencil._ey, _stencil._ez}, 0);

  for (int ic = 0; ic < _stencil._q; ic++)
  {
    auto exyz_ic = e_xyz.index({Slice(), ic}).flatten();
    torch::Tensor ccr = torch::outer(exyz_ic, exyz_ic).flatten();
    fneqtimescc += (fneq.select(1, ic).view({nx * ny * nz, 1}) * ccr.view({1, _stencil._q}));
  }

  // Compute Hermite tensor
  torch::Tensor H2 = torch::zeros({1, _stencil._q}, MooseTensor::floatTensorOptions());
  for (int ic = 0; ic < _stencil._q; ic++)
  {
    auto exyz_ic = e_xyz.index({Slice(), ic}).flatten();
    torch::Tensor ccr = torch::outer(exyz_ic, exyz_ic) / _lb_problem._cs2 -
                        torch::eye(3, MooseTensor::floatTensorOptions());
    H2 = ccr.flatten().unsqueeze(0).expand({nx * ny * nz, _stencil._q});

    // Compute regularized non-equilibrium distribution
    f_neq_hat.index_put_(
        {Slice(), ic},
        (_stencil._weights[ic] * (1.0 / (2.0 * _lb_problem._cs2)) * (fneqtimescc * H2).sum(1)));
  }

  _fneq = f_neq_hat.view({nx, ny, nz, _stencil._q});
}

template <>
void
LBMCollisionDynamicsTempl<0>::BGKDynamics()
{
  /* LBM BGK collision */
  _u = _feq + _fneq - 1.0 / _tau_0 * _fneq;
  _lb_problem.maskedFillSolids(_u, 0);
}

template <>
void
LBMCollisionDynamicsTempl<1>::MRTDynamics()
{
  /* LBM MRT collision */
  const auto shape = _u.sizes();

  // f = M^{-1} x S x M x (f - feq)
  _u = _feq + _fneq -
       torch::matmul(_stencil._M_inv,
                     torch::matmul(_stencil._S,
                                   torch::matmul(_stencil._M, (_fneq).view({-1, _stencil._q}).t())))
           .t()
           .view({shape});

  _lb_problem.maskedFillSolids(_u, 0);
}

template <>
void
LBMCollisionDynamicsTempl<2>::SmagorinskyDynamics()
{
  int64_t nx = _shape[0];
  int64_t ny = _shape[1];
  int64_t nz = _shape[2];

  auto f_neq_hat = _fneq.view({nx * ny * nz, _stencil._q, 1, 1, 1});

  auto zeros = torch::zeros({_stencil._q}, MooseTensor::intTensorOptions());
  auto ones = torch::ones({_stencil._q}, MooseTensor::intTensorOptions());

  auto ex_2d = torch::stack({_stencil._ex, zeros, zeros});
  auto ey_2d = torch::stack({zeros, _stencil._ey, zeros});
  auto ez_2d = torch::stack({zeros, zeros, _stencil._ez});

  if (nz == 1)
    ez_2d = torch::stack({ones, zeros, _stencil._ez});

  // outer product
  // expected shape: _q, 3, 3, 3
  auto outer_products = torch::zeros({_stencil._q, 3, 3, 3}, MooseTensor::intTensorOptions());

  for (int i = 0; i < _stencil._q; i++)
  {
    auto ex_col = ex_2d.index({Slice(), i});
    auto ey_col = ey_2d.index({Slice(), i});
    auto ez_col = ez_2d.index({Slice(), i});
    auto outer_product = torch::einsum("i,j,k->kij", {ex_col, ey_col, ez_col});
    outer_products[i] = outer_product;
  }
  outer_products = outer_products.view({1, _stencil._q, 3, 3, 3});

  // momentum flux
  auto Q = torch::sum(f_neq_hat * outer_products, 1).view({nx, ny, nz, 3, 3, 3});

  // mean density
  _mean_density = torch::mean(torch::sum(_f, 3)).item<double>();

  // Frobenius norm
  auto Q_mean = torch::norm(Q, 2, {3, 4, 5}) / (_mean_density * _lb_problem._cs2);

  // subgrid time scale factor
  auto t_sgs = sqrt(_C_s) * _delta_x / _lb_problem._cs;
  auto eta = _tau_0 / t_sgs;

  // mean strain rate
  auto S = (-1.0 * eta + torch::sqrt(eta * eta + 4.0 * Q_mean)) / (2.0 * t_sgs);

  // relaxation parameter
  // torch::Tensor tau_eff =
  //     0.5 * (_tau_0 + torch::sqrt(Q_mean * 1.0 / (_mean_density * _lb_problem._cs4) * 2.0 *
  //                                     sqrt(2.0) * _C_s * _delta_x * _delta_x +
  //                                 _tau_0 * _tau_0));
  auto tau_eff = _tau_0 + _C_s * _delta_x * _delta_x * S / _lb_problem._cs2;
  tau_eff.unsqueeze_(3);

  // BGK collision
  _u = _feq + _fneq - 1.0 / tau_eff * _fneq;

  _lb_problem.maskedFillSolids(_u, 0);
}

template <int coll_dyn>
void
LBMCollisionDynamicsTempl<coll_dyn>::computeBuffer()
{
  if (_projection)
    HermiteRegularization();
  else
    _fneq = _f - _feq;

  switch (coll_dyn)
  {
    case 0:
      BGKDynamics();
      break;
    case 1:
      MRTDynamics();
      break;
    case 2:
      SmagorinskyDynamics();
      break;
    default:
      mooseError("Undefined template value");
  }
  _lb_problem.maskedFillSolids(_u, 0);
}

template class LBMCollisionDynamicsTempl<0>;
template class LBMCollisionDynamicsTempl<1>;
template class LBMCollisionDynamicsTempl<2>;
