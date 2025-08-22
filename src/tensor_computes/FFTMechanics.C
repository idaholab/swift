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
#include <util/Optional.h>
#include <numeric>

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
  params.addParam<unsigned int>("l_max_its",
                                "Maximum number of congugate gradient solve iterations");
  params.addParam<Real>("nl_rel_tol", 1e-5, "Nonlinear solve absolute tolerance");
  params.addParam<Real>("nl_abs_tol", 1e-8, "Nonlinear solve relative tolerance");
  params.addParam<unsigned int>("nl_max_its", 100, "Maximum number of nonlinear solve iterations");
  params.addParam<TensorInputBufferName>("stress", "stress", "Computed stress");
  params.addParam<TensorInputBufferName>("tangent_operator", "dstressdstrain", "Tangent operator");
  params.addRequiredParam<TensorComputeName>("constitutive_model",
                                             "Tensor compute for the constitutive model (computes "
                                             "stress from displacement gradeint tensor)");
  params.addParam<TensorInputBufferName>("applied_macroscopic_strain",
                                         "Applied macroscopic strain");
  params.addParam<TensorInputBufferName>("F", "F", "Deformation gradient tensor.");
  params.addParam<unsigned int>("hutchinson_steps",
                                0,
                                "Steps for diagonal estimation with Hutchinson's method used in "
                                "Jacobi preconditioning. 0 skips preconditioning.");
  params.addParam<bool>("block_jacobi",
                        false,
                        "Use block-Jacobi (local compliance) preconditioner instead of diagonal.");
  params.addParam<Real>("block_jacobi_damp",
                        1e-8,
                        "Relative damping added to local tangent blocks before inversion.");
  params.addParam<Real>(
      "jacobi_min_rel",
      1e-3,
      "Minimum relative floor for stochastic Jacobi diagonal (relative to median).");
  params.addParam<Real>(
      "jacobi_inv_cap", 0.0, "Cap on inverse diagonal scaling; 0 disables clamping.");
  params.addParam<bool>("verbose", false, "Print non-linear residuals.");
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
    _tF(getInputBuffer<>("F")),
    _tP(getInputBuffer("stress")),
    _tK4(getInputBuffer("tangent_operator")),
    _l_tol(getParam<Real>("l_tol")),
    _l_max_its(isParamValid("l_max_its") ? getParam<unsigned int>("l_max_its")
                                         : _domain.getNumberOfCells()),
    _nl_rel_tol(getParam<Real>("nl_rel_tol")),
    _nl_abs_tol(getParam<Real>("nl_abs_tol")),
    _nl_max_its(getParam<unsigned int>("nl_max_its")),
    _constitutive_model(getCompute("constitutive_model")),
    _applied_macroscopic_strain(isParamValid("applied_macroscopic_strain")
                                    ? &getInputBuffer("applied_macroscopic_strain")
                                    : nullptr),
    _hutchinson_steps(getParam<unsigned int>("hutchinson_steps")),
    _block_jacobi(getParam<bool>("block_jacobi")),
    _block_jacobi_damp(getParam<Real>("block_jacobi_damp")),
    _jacobi_min_rel(getParam<Real>("jacobi_min_rel")),
    _jacobi_inv_cap(getParam<Real>("jacobi_inv_cap")),
    _verbose(getParam<bool>("verbose"))
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
FFTMechanics::check()
{
  const auto & stress_name = getParam<TensorOutputBufferName>("stress");
  if (!_constitutive_model.getSuppliedItems().count(stress_name))
    paramError("constitutive_model", "does not provide stress tensor '", stress_name, "'.");
}

void
FFTMechanics::computeBuffer()
{
  using namespace MooseTensor;

  const auto shape = _domain.getShape();
  std::vector<int64_t> _r2shape(shape.begin(), shape.end());
  _r2shape.push_back(_dim);
  _r2shape.push_back(_dim);

  const auto G = [&](const torch::Tensor & A2)
  { return _domain.ifft(ddot42(_Ghat4, _domain.fft(A2))).reshape(-1); };
  const auto K_dF = [&](const torch::Tensor & dFm)
  { return trans2(ddot42(_tK4, trans2(dFm.reshape(_r2shape)))); };
  const auto G_K_dF = [&](const torch::Tensor & dFm) { return G(K_dF(dFm)); };

  // initialize deformation gradient, and stress/stiffness       [grid of tensors]
  _u = _tF;
  _constitutive_model.computeBuffer();

  // initial residual: distribute "barF" over grid using "K4"
  auto b = _applied_macroscopic_strain ? -G_K_dF(_applied_macroscopic_strain->expand(_r2_shape))
                                       : -G_K_dF(torch::zeros_like(_tF));

  // _u += DbarF;
  if (_applied_macroscopic_strain)
    _u = _u + _applied_macroscopic_strain->expand(_r2_shape);

  const auto Fn =
      at::linalg_norm(_u, c10::nullopt, c10::nullopt, false, c10::nullopt).cpu().item<double>();

  auto dFm = torch::zeros_like(b);

  // iterate as long as the iterative update does not vanish
  for (const auto iiter : make_range(_nl_max_its))
  {
    c10::optional<torch::Tensor> invK4;
    const auto diag_estimate =
        (!_block_jacobi && _hutchinson_steps)
            ? torch::abs(estimateJacobiPreconditioner(G_K_dF, b, _hutchinson_steps))
            : torch::ones_like(b);
    auto inv_diag = torch::ones_like(b);
    if (!_block_jacobi && _hutchinson_steps)
    {
      // Robust floor relative to a nonzero scale to avoid huge inverse scaling
      auto mask = diag_estimate > 1e-9; // ignore near-zero estimates
      auto selected = at::masked_select(diag_estimate, mask);
      auto scale_t = selected.numel() > 0 ? selected.mean() : diag_estimate.mean();
      auto floor_t = scale_t * _jacobi_min_rel;
      auto diag_precond = torch::clamp(diag_estimate, floor_t, c10::nullopt);
      inv_diag = 1.0 / diag_precond;
      if (_jacobi_inv_cap > 0.0)
      {
        inv_diag = torch::clamp(inv_diag, 0.0, _jacobi_inv_cap);
      }
    }
    const auto M_inv = [&](const torch::Tensor & x)
    {
      if (_block_jacobi)
      {
        if (!invK4.has_value())
          invK4 = MooseTensor::invertLocalBlocksDamped(_tK4, _block_jacobi_damp);
        auto x2 = x.reshape(_r2_shape);
        auto z2raw = MooseTensor::trans2(MooseTensor::ddot42(*invK4, MooseTensor::trans2(x2)));
        // Enforce zero-mean (remove k=0 mode) without FFT cost
        std::vector<int64_t> reduce_dims(z2raw.dim() - 2);
        std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
        auto mean2 = z2raw.mean(reduce_dims, /*keepdim=*/true);
        auto z2 = z2raw - mean2;
        return z2.reshape(-1);
      }
      else
      {
        auto x2 = x.reshape(_r2_shape);
        auto z2raw = x2 * inv_diag.reshape(_r2_shape);
        // Enforce zero-mean (remove k=0 mode) without FFT cost
        std::vector<int64_t> reduce_dims(z2raw.dim() - 2);
        std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
        auto mean2 = z2raw.mean(reduce_dims, /*keepdim=*/true);
        auto z2 = z2raw - mean2;
        return z2.reshape(-1);
      }
    };

    const auto [dFm_new, iterations, lnorm] =
        (_block_jacobi || _hutchinson_steps)
            ? conjugateGradientSolve(G_K_dF, b, dFm, _l_tol, _l_max_its, M_inv)
            : conjugateGradientSolve(G_K_dF, b, dFm, _l_tol, _l_max_its);
    dFm = dFm_new;

    // update DOFs (array -> tens.grid)
    _u += dFm.reshape(_r2_shape);

    // new residual stress and tangent
    _constitutive_model.computeBuffer();

    // convert res.stress to residual
    b = -G(_tP);

    const auto anorm =
        at::linalg_norm(dFm, c10::nullopt, c10::nullopt, false, c10::nullopt).cpu().item<double>();
    const auto rnorm = anorm / Fn;

    // print nonlinear residual to the screen
    if (_verbose)
      Moose::out << iiter << " |R|=" << anorm << "\t|R/R0|=" << rnorm << std::endl;

    // check convergence
    if (rnorm < _nl_rel_tol || anorm < _nl_abs_tol)
      return;
  }

  paramError("nl_max_its",
             "Exceeded the maximum number of nonlinear iterations without converging.");
}
