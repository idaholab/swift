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

void
printBuffer(const torch::Tensor & t, const unsigned int & precision, const unsigned int & index);

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
