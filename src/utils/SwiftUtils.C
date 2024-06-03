//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "SwiftUtils.h"
#include "SwiftApp.h"
#include "Moose.h"

namespace MooseFFT
{

torch::Device
getDevice()
{
  return torch::cuda::is_available() && !forceCPU() ? torch::kCUDA : torch::kCPU;
}

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

const torch::TensorOptions
floatTensorOptions()
{
  return torch::TensorOptions()
      .dtype(torch::kFloat64)
      .layout(torch::kStrided)
      .memory_format(torch::MemoryFormat::Contiguous)
      .pinned_memory(false)
      .device(getDevice())
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
      .device(getDevice())
      .requires_grad(false);
}

} // namespace MooseFFT
