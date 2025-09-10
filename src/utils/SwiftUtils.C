/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "SwiftUtils.h"
#include "SwiftApp.h"
#include "MooseUtils.h"
#include "Moose.h"
// for batched linear algebra
#include <ATen/ops/linalg_inv.h>
#include <ATen/ops/linalg_pinv.h>

namespace MooseTensor
{

struct TorchDeviceSingleton
{
  static bool isSupported(torch::Dtype dtype, torch::Device device)
  {
    try
    {
      auto tensor = torch::zeros({1}, torch::dtype(dtype).device(device));
      return true;
    }
    catch (const std::exception &)
    {
      return false;
    }
  }

  TorchDeviceSingleton()
    : _device_string(torchDevice().empty() ? (torch::cuda::is_available()
                                                  ? "cuda"
                                                  : (torch::mps::is_available() ? "mps" : "cpu"))
                                           : torchDevice()),
      _device(_device_string),
      _floating_precision(precision().empty() ? "DEVICE_DEFAULT" : precision()),
      _float_dtype(_floating_precision == "DEVICE_DEFAULT" || _floating_precision == "DOUBLE"
                       ? (isSupported(torch::kFloat64, _device) ? torch::kFloat64 : torch::kFloat32)
                       : torch::kFloat32),
      _complex_float_dtype(isSupported(torch::kComplexDouble, _device) ? torch::kComplexDouble
                                                                       : torch::kComplexFloat),
      _int_dtype(isSupported(torch::kInt64, _device) ? torch::kInt64 : torch::kInt32)
  {
    mooseInfo("Running on '", _device_string, "'.");
    if (_float_dtype == torch::kFloat64)
      mooseInfo("Device supports double precision floating point numbers.");
    else
      mooseWarning("Running with single precision floating point numbers");
  }

  const std::string _device_string;
  const torch::Device _device;
  const std::string _floating_precision;
  const torch::Dtype _float_dtype;
  const torch::Dtype _complex_float_dtype;
  const torch::Dtype _int_dtype;
};

void
printTensorInfo(const torch::Tensor & x)
{
  Moose::out << "      dimension: " << x.dim() << std::endl;
  Moose::out << "          shape: " << x.sizes() << std::endl;
  Moose::out << "          dtype: " << x.dtype() << std::endl;
  Moose::out << "         device: " << x.device() << std::endl;
  Moose::out << "  requires grad: " << (x.requires_grad() ? "true" : "false") << std::endl;
  Moose::out << std::endl;
}

void
printTensorInfo(const std::string & name, const torch::Tensor & x)
{
  Moose::out << "============== " << name << " ==============\n";
  printTensorInfo(x);
  Moose::out << std::endl;
}

void
printElementZero(const torch::Tensor & tensor)
{
  // Access the element at all zero indices
  auto element = tensor[0][0];
  // for (int i = 1; i < tensor.dim(); ++i)
  //   element = element[0];

  Moose::out << element << std::endl;
}

void
printElementZero(const std::string & name, const torch::Tensor & x)
{
  Moose::out << "============== " << name << " ==============\n";
  printElementZero(x);
  Moose::out << std::endl;
}

const torch::TensorOptions
floatTensorOptions()
{
  const static TorchDeviceSingleton ts;
  return torch::TensorOptions()
      .dtype(ts._float_dtype)
      .layout(torch::kStrided)
      .memory_format(torch::MemoryFormat::Contiguous)
      .pinned_memory(false)
      .device(ts._device)
      .requires_grad(false);
}

const torch::TensorOptions
complexFloatTensorOptions()
{
  const static TorchDeviceSingleton ts;
  return torch::TensorOptions()
      .dtype(ts._complex_float_dtype)
      .layout(torch::kStrided)
      .memory_format(torch::MemoryFormat::Contiguous)
      .pinned_memory(false)
      .device(ts._device)
      .requires_grad(false);
}

const torch::TensorOptions
intTensorOptions()
{
  const static TorchDeviceSingleton ts;
  return torch::TensorOptions()
      .dtype(ts._int_dtype)
      .layout(torch::kStrided)
      .memory_format(torch::MemoryFormat::Contiguous)
      .pinned_memory(false)
      .device(ts._device)
      .requires_grad(false);
}

torch::Tensor
unsqueeze0(const torch::Tensor & t, unsigned int ndim)
{
  torch::Tensor u = t;
  for (unsigned int i = 0; i < ndim; ++i)
    u = u.unsqueeze(0);
  return u;
}

torch::Tensor
trans2(const torch::Tensor & A2)
{
  return torch::einsum("...ij          ->...ji  ", {A2});
}

torch::Tensor
ddot42(const torch::Tensor & A4, const torch::Tensor & B2)
{
  return torch::einsum("...ijkl,...lk  ->...ij  ", {A4, B2});
}

torch::Tensor
ddot44(const torch::Tensor & A4, const torch::Tensor & B4)
{
  return torch::einsum("...ijkl,...lkmn->...ijmn", {A4, B4});
}

torch::Tensor
dot22(const torch::Tensor & A2, const torch::Tensor & B2)
{
  return torch::einsum("...ij  ,...jk  ->...ik  ", {A2, B2});
}

torch::Tensor
dot24(const torch::Tensor & A2, const torch::Tensor & B4)
{
  return torch::einsum("...ij  ,...jkmn->...ikmn", {A2, B4});
}

torch::Tensor
dot42(const torch::Tensor & A4, const torch::Tensor & B2)
{
  return torch::einsum("...ijkl,...lm  ->...ijkm", {A4, B2});
}

torch::Tensor
dyad22(const torch::Tensor & A2, const torch::Tensor & B2)
{
  return torch::einsum("...ij  ,...kl  ->...ijkl", {A2, B2});
}

void
printBuffer(const torch::Tensor & t, const unsigned int & precision, const unsigned int & index)
{
  /**
   * Print the entire field for debugging
   */
  torch::Tensor field = t;
  // for buffers higher than 3 dimensions, such as distribution functions
  // pass an index to print or call this method repeatedly to print all directions
  // higher than 4 dimensions is not supported

  if (t.dim() == 4)
    field = t.select(3, index);

  if (t.dim() > 4)
    mooseError("Higher than 4 dimensional tensor buffers are not supported.");

  if (t.dim() == 2)
  {
    for (int64_t j = 0; j < field.size(1); j++)
    {
      for (int64_t k = 0; k < field.size(0); k++)
        std::cout << std::fixed << std::setprecision(precision) << field[k][j].item<Real>() << " ";
      std::cout << std::endl;
    }
  }

  else if (t.dim() >= 3)
  {
    for (int64_t i = 0; i < field.size(2); i++)
    {
      for (int64_t j = 0; j < field.size(1); j++)
      {
        for (int64_t k = 0; k < field.size(0); k++)
          std::cout << std::fixed << std::setprecision(precision) << field[k][j][i].item<Real>()
                    << " ";
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }

  else if (t.dim() == 1)
  {
    for (int64_t k = 0; k < field.size(0); k++)
      std::cout << std::fixed << std::setprecision(precision) << field[k].item<Real>() << " ";
    std::cout << std::endl;
  }

  else
    mooseError("Unsupported output dimension");
}
// Invert local 4th-order blocks (batch of d*d x d*d matrices)
torch::Tensor
invertLocalBlocks(const torch::Tensor & K4)
{
  // Expect shape: [..., d, d, d, d]
  const auto d = K4.size(-1);
  // Flatten last 4 dims to (d*d, d*d) with batch = prod(leading dims)
  auto K2 = K4.reshape({-1, d * d, d * d});
  // Batched inverse
  auto K2_inv = at::linalg_inv(K2);
  // Restore original shape
  return K2_inv.reshape(K4.sizes());
}

torch::Tensor
invertLocalBlocksDamped(const torch::Tensor & K4, double damp_rel)
{
  // Flatten to (batch, n, n)
  const auto d = K4.size(-1);
  const auto n = d * d;
  auto K2 = K4.reshape({-1, n, n});

  // Build batched identity
  auto I = torch::eye(n, K4.options()).unsqueeze(0).expand({K2.size(0), n, n});

  // Scale damping by mean absolute diagonal across batch
  auto diag = K2.diagonal(0, -2, -1);
  double scale = diag.abs().mean().template item<double>();
  if (!(scale > 0.0))
    scale = 1.0;
  const double eps = damp_rel > 0.0 ? damp_rel * scale : 0.0;

  auto K2_reg = eps > 0.0 ? (K2 + eps * I) : K2;

  try
  {
    auto K2_inv = at::linalg_inv(K2_reg);
    return K2_inv.reshape(K4.sizes());
  }
  catch (const c10::Error &)
  {
    auto K2_pinv = at::linalg_pinv(K2_reg);
    return K2_pinv.reshape(K4.sizes());
  }
}

// Diagonal estimation with Hutchinson's method
torch::Tensor
estimateJacobiPreconditioner(const std::function<torch::Tensor(const torch::Tensor &)> & A,
                             const torch::Tensor & template_vec,
                             int num_samples)
{
  torch::Tensor diag_est = torch::zeros_like(template_vec);

  for (int i = 0; i < num_samples; ++i)
  {
    torch::Tensor v = torch::randint(0, 2, template_vec.sizes(), template_vec.options()) * 2 - 1;
    torch::Tensor Av = A(v);
    diag_est += v * Av;
  }

  diag_est /= num_samples;

  // Avoid division by zero
  const torch::Tensor eps = torch::tensor(1e-10, diag_est.options());
  return torch::where(torch::abs(diag_est) < eps, eps, diag_est);
}

std::tuple<torch::Tensor, unsigned int, double>
conjugateGradientSolve(const std::function<torch::Tensor(const torch::Tensor &)> & A,
                       torch::Tensor b,
                       torch::Tensor x0,
                       double tol,
                       int64_t maxiter,
                       const std::function<torch::Tensor(const torch::Tensor &)> & M)
{
  // initialize solution guess
  torch::Tensor x = x0.defined() ? x0.clone() : torch::zeros_like(b);

  // norm of b (for relative tolerance)
  const double b_norm = torch::norm(b).cpu().template item<double>();
  if (b_norm == 0.0)
    // solution is zero if b is zero
    return {x, 0u, 0.0};

  // default max iterations
  if (!maxiter)
    maxiter = b.numel();

  // initial residual
  torch::Tensor r = b - A(x);

  // Apply preconditioner (or identity)
  torch::Tensor z = M(r); // z = M^{-1} r

  // initial search direction p
  torch::Tensor p = z.clone();

  // dot product (r, z)
  double rz_old = torch::sum(r * z).cpu().template item<double>();

  // CG iteration
  double res_norm;
  for (const auto k : libMesh::make_range(maxiter))
  {
    // compute matrix-vector product
    const auto Ap = A(p);

    // step size alpha
    double alpha = rz_old / torch::sum(p * Ap).cpu().template item<double>();

    // update solution
    x = x + alpha * p;

    // update residual
    r = r - alpha * Ap;
    res_norm = torch::norm(r).cpu().template item<double>(); // ||r||

    // std::cout << k << ' '<< res_norm << '\n';

    // Converged to desired tolerance
    if (res_norm <= tol * b_norm)
      return {x, k + 1, res_norm};

    // apply preconditioner to new residual
    z = M(r);
    const auto rz_new = torch::sum(r * z).cpu().template item<double>();

    // update scalar beta
    double beta = rz_new / rz_old;

    // update search direction
    p = z + beta * p;

    // prepare for next iteration
    rz_old = rz_new;
  }

  // Reached max iterations without full convergence
  return {x, maxiter, res_norm};
}

} // namespace MooseTensor
