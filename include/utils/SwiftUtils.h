/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include <torch/torch.h>
#include "libmesh/int_range.h"

#define pti(tensor) MooseTensor::printTensorInfo(#tensor, tensor)

#define pez(tensor) MooseTensor::printElementZero(#tensor, tensor)

namespace MooseTensor
{

/// Passkey pattern key template
template <typename T>
class Key
{
  friend T;
  Key() {}
  Key(Key const &) {}
};

void printTensorInfo(const torch::Tensor & x);
void printTensorInfo(const std::string & name, const torch::Tensor & x);

void printElementZero(const torch::Tensor & tensor);
void printElementZero(const std::string & name, const torch::Tensor & tensor);

void printBuffer(const torch::Tensor & t,
                 const unsigned int & precision = 5,
                 const unsigned int & index = 0);

const torch::TensorOptions floatTensorOptions();
const torch::TensorOptions complexFloatTensorOptions();
const torch::TensorOptions intTensorOptions();

/// unsqueeze(0) ndim times
torch::Tensor unsqueeze0(const torch::Tensor & t, unsigned int ndim);

torch::Tensor trans2(const torch::Tensor & A2);
torch::Tensor ddot42(const torch::Tensor & A4, const torch::Tensor & B2);
torch::Tensor ddot44(const torch::Tensor & A4, const torch::Tensor & B4);
torch::Tensor dot22(const torch::Tensor & A2, const torch::Tensor & B2);
torch::Tensor dot24(const torch::Tensor & A2, const torch::Tensor & B4);
torch::Tensor dot42(const torch::Tensor & A4, const torch::Tensor & B2);
torch::Tensor dyad22(const torch::Tensor & A2, const torch::Tensor & B2);

// Invert local 4th-order blocks (...., i, j, k, l) -> (...., i, j, k, l)
// Treats each grid point's (i,j)-(k,l) matrix as a (d*d x d*d) block and inverts it in batch.
// Returns a tensor with the same shape as the input, containing per-point block inverses.
torch::Tensor invertLocalBlocks(const torch::Tensor & K4);

// Damped inversion of local 4th-order blocks with optional pinv fallback.
// damp_rel scales an identity added to each (d*d x d*d) block by damp_rel * mean(|diag|).
// If inversion fails, falls back to pseudo-inverse.
torch::Tensor invertLocalBlocksDamped(const torch::Tensor & K4, double damp_rel = 1e-8);

torch::Tensor
estimateJacobiPreconditioner(const std::function<torch::Tensor(const torch::Tensor &)> & A,
                             const torch::Tensor & template_vec,
                             int num_samples = 6);

std::tuple<torch::Tensor, unsigned int, double>

conjugateGradientSolve(const std::function<torch::Tensor(const torch::Tensor &)> & A,
                       torch::Tensor b,
                       torch::Tensor x0,
                       double tol,
                       int64_t maxiter,
                       const std::function<torch::Tensor(const torch::Tensor &)> & M);

template <typename T>
std::tuple<torch::Tensor, unsigned int, double>
conjugateGradientSolve(
    T A, torch::Tensor b, torch::Tensor x0 = {}, double tol = 1e-6, int64_t maxiter = 0)
{
  return conjugateGradientSolve(A, b, x0, tol, maxiter, [](const torch::Tensor r) { return r; });
}

} // namespace MooseTensor
