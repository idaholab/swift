/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include <torch/torch.h>

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

const torch::TensorOptions floatTensorOptions();
const torch::TensorOptions complexFloatTensorOptions();
const torch::TensorOptions intTensorOptions();

void read2DStructuredLBMMeshFromVTK(const std::string& filePath,
                      torch::Tensor& binaryMedia,
                      torch::Tensor& poreSize,
                      torch::Tensor& knudsenNumber,
                      const std::vector<int> &dims);

} // namespace MooseTensor

