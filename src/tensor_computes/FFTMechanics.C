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
    _tP(getOutputBuffer("stress"))
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
                   DbarF.index({torch::indexing::Ellipsis, 0, 1}) + 1.0);

  DbarF = DbarF.expand(_r2_shape);

  // initial residual: distribute "barF" over grid using "K4"
  auto b = -G_K_dF(DbarF);
  // _u += DbarF;
  _u = _u + DbarF;

  const auto Fn = torch::linalg::norm(_u, c10::nullopt, c10::nullopt, false, c10::nullopt);
  unsigned int iiter = 0;
  auto dFm = torch::zeros_like(b);

  // iterate as long as the iterative update does not vanish
  while (true)
  {
    const auto [dFm_new, iterations, lnorm] = conjugateGradientSolve(G_K_dF, b, dFm, 1e-2);
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

    const double nnorm =
        (torch::linalg::norm(dFm, c10::nullopt, c10::nullopt, false, c10::nullopt) / Fn)
            .cpu()
            .item<double>();
    std::cout << nnorm << '\n'; // print residual to the screen

    // check convergence
    if (nnorm < 1.e-5 && iiter > 0)
      break;

    iiter++;
  }
}

#endif
