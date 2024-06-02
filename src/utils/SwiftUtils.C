//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "SwiftUtils.h"

namespace MooseFFT
{

void
printTensorInfo(const torch::Tensor & x)
{
  std::cout << "      dimension: " << x.dim() << std::endl;
  std::cout << "          shape: " << x.sizes() << std::endl;
  std::cout << "          dtype: " << x.dtype() << std::endl;
  std::cout << "         device: " << x.device() << std::endl;
  std::cout << "  requires grad: " << (x.requires_grad() ? "true" : "false") << std::endl;
  std::cout << std::endl;
}

const torch::TensorOptions
floatTensorOptions()
{
  return torch::TensorOptions()
      .dtype(torch::kFloat64)
      .layout(torch::kStrided)
      .memory_format(torch::MemoryFormat::Contiguous)
      .pinned_memory(false)
      .device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
      .requires_grad(false);
}

const torch::TensorOptions
intTensorOptions()
{
  return torch::TensorOptions()
      .dtype(torch::kInt64)
      .layout(torch::kStrided)
      .memory_format(torch::MemoryFormat::Contiguous)
      .pinned_memory(false)
      .device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
      .requires_grad(false);
}

} // namespace MooseFFT
