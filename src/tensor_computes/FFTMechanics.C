/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "FFTMechanics.h"
#include "DomainAction.h"
#include "MooseError.h"
#include "SwiftUtils.h"
#include <ATen/TensorIndexing.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/unsqueeze_ops.h>
#include <torch/linalg.h>
#include <util/Optional.h>

#ifdef NEML2_ENABLED

registerMooseObject("SwiftApp", FFTMechanics);

InputParameters
FFTMechanics::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("deGeus variational mechanics solve. Updates the coupled buffer "
                             "holding the deformation gradient tensor.");
  // params.addParam<TensorInputBufferName>("C0", "Stiffness tensor estimate");
  // params.addParam<TensorInputBufferName>("C", "Stiffness tensor");
  params.addRequiredParam<TensorInputBufferName>("K", "Bulk modulus");
  params.addParam<TensorInputBufferName>("mu", "Shear modulus");
  params.addParam<Real>("l_tol", 1e-2, "Linear congugate gradient solve tolerance");
  params.addParam<unsigned int>(
      "l_max_its", 100, "Maximum number of congugate gradient solve iterations");
  params.addParam<Real>("nl_rel_tol", 1e-5, "Nonlinear solve absolute tolerance");
  params.addParam<Real>("nl_abs_tol", 1e-8, "Nonlinear solve relative tolerance");
  params.addParam<unsigned int>("nl_max_its", 100, "Maximum number of nonlinear solve iterations");
  params.addParam<TensorOutputBufferName>("stress", "stress", "Computed stress");
  return params;
}

FFTMechanics::FFTMechanics(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _ti(torch::eye(_dim, MooseTensor::floatTensorOptions())),
    _tI(MooseTensor::unsqueeze0(_ti, _dim)),
    _tI4(MooseTensor::unsqueeze0(torch::einsum("il,jk", {_ti, _ti}), _dim)),
    _tI4rt(MooseTensor::unsqueeze0(torch::einsum("ik,jl", {_ti, _ti}), _dim)),
    _tI4s((_tI4 + _tI4rt) / 2.0),
    _tII(MooseTensor::dyad22(_tI, _tI)),
    _tK(getInputBuffer("K")),
    _tmu(getInputBuffer("mu")),
    _r2_shape(_domain.getValueShape({_dim, _dim})),
    _tP(getOutputBuffer("stress")),
    _l_tol(getParam<Real>("l_tol")),
    _l_max_its(getParam<unsigned int>("l_max_its")),
    _nl_rel_tol(getParam<Real>("nl_rel_tol")),
    _nl_abs_tol(getParam<Real>("nl_abs_tol")),
    _nl_max_its(getParam<unsigned int>("nl_max_its"))
{
  // Build projection tensor once
  const auto & q = _domain.getKGrid();
  const auto Q = _domain.getKSquare().unsqueeze(-1).unsqueeze(-1);

  auto M = torch::where(Q == 0, 0.0, q.unsqueeze(-2) * q.unsqueeze(-1) / Q);

  M = M.unsqueeze(-3).unsqueeze(-1);

  const auto delta_im = _ti.unsqueeze(1).unsqueeze(1).expand({_dim, _dim, _dim, _dim});

  _Ghat4 = (M * delta_im).to(MooseTensor::complexFloatTensorOptions());
}

void
FFTMechanics::computeBuffer()
{
  using namespace MooseTensor;
  torch::Tensor K4;

  const auto shape = _domain.getShape();
  std::vector<int64_t> _r2shape(shape.begin(), shape.end());
  _r2shape.push_back(_dim);
  _r2shape.push_back(_dim);

  const auto G = [&](const torch::Tensor & A2)
  { return _domain.ifft(ddot42(_Ghat4, _domain.fft(A2))).reshape(-1); };
  const auto K_dF = [&](const torch::Tensor & dFm)
  { return trans2(ddot42(K4, trans2(dFm.reshape(_r2shape)))); };
  const auto G_K_dF = [&](const torch::Tensor & dFm) { return G(K_dF(dFm)); };

  // constitutive model: grid of "F" -> grid of "P", "K4"        [grid of tensors]
  auto constitutive = [&](const torch::Tensor & F)
  {
    const auto C4 =
        _tK.reshape(_domain.getValueShape({1, 1, 1, 1})) * _tII +
        2. * _tmu.reshape(_domain.getValueShape({1, 1, 1, 1})) * (_tI4s - 1. / 3. * _tII);
    const auto S = ddot42(C4, .5 * (dot22(trans2(F), F) - _tI));
    const auto P = dot22(F, S);
    const auto K4 = dot24(S, _tI4) + ddot44(ddot44(_tI4rt, dot42(dot24(F, C4), trans2(F))), _tI4rt);
    return std::make_pair(P, K4);
  };

  // initialize deformation gradient, and stress/stiffness       [grid of tensors]
  _u = _tI.clone();
  const auto P_K4 = constitutive(_u);
  _tP = P_K4.first;
  K4 = P_K4.second;

  // set macroscopic loading
  auto DbarF = torch::zeros_like(_tI);
  // DbarF[..., 0, 1] += 1.0;
  DbarF.index_put_({torch::indexing::Ellipsis, 0, 1},
                   DbarF.index({torch::indexing::Ellipsis, 0, 1}) + _time);

  DbarF = DbarF.expand(_r2_shape);

  // initial residual: distribute "barF" over grid using "K4"
  auto b = -G_K_dF(DbarF);
  // _u += DbarF;
  _u = _u + DbarF;

  const auto Fn =
      at::linalg_norm(_u, c10::nullopt, c10::nullopt, false, c10::nullopt).cpu().item<double>();

  unsigned int iiter = 0;
  auto dFm = torch::zeros_like(b);

  // iterate as long as the iterative update does not vanish
  while (true)
  {
    const auto [dFm_new, iterations, lnorm] =
        conjugateGradientSolve(G_K_dF, b, dFm, _l_tol, _l_max_its);
    dFm = dFm_new;

    // update DOFs (array -> tens.grid)
    // _u += dFm.reshape(_r2_shape);
    _u = _u + dFm.reshape(_r2_shape);

    // new residual stress and tangent
    const auto P_K4 = constitutive(_u);
    _tP = P_K4.first;
    K4 = P_K4.second;

    // convert res.stress to residual
    b = -G(_tP);

    const auto anorm =
        at::linalg_norm(dFm, c10::nullopt, c10::nullopt, false, c10::nullopt).cpu().item<double>();
    const auto rnorm = anorm / Fn;

    std::cout << anorm << ' ' << rnorm << '\n'; // print residual to the screen

    // check convergence
    if ((rnorm < _nl_rel_tol || anorm < _nl_abs_tol) && iiter > 0)
      break;

    iiter++;

    if (iiter > _nl_max_its)
      paramError("nl_max_its",
                 "Exceeded the maximum number of nonlinear iterations without converging.");
  }
}

#endif
