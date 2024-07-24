//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifdef NEML2_ENABLED

#include "SwiftUtils.h"
#include "SwiftApp.h"
#include "MooseUtils.h"
#include "Moose.h"

namespace MooseTensor
{

struct TorchDeviceSingleton
{
  static bool isSupported(torch::Dtype dtype, torch::Device device)
  {
    try
    {
      auto tensor = torch::rand({1}, torch::dtype(dtype).device(device));
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
      _float_dtype(isSupported(torch::kFloat64, _device) ? torch::kFloat64 : torch::kFloat32),
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
  const torch::Dtype _float_dtype;
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

} // namespace MooseTensor

#endif
