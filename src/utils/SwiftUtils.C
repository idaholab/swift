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
printTensorInfo(const std::string & name,const torch::Tensor & x)
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

void
read2DStructuredLBMMeshFromVTK(const std::string& filePath,
                      torch::Tensor& binaryMedia,
                      torch::Tensor& poreSize,
                      torch::Tensor& knudsenNumber,
                      const std::vector<int> &dims)
{
  #ifdef SWIFT_HAVE_VTK
    /*
      Function to read binary porous media, local pore size distribution and Knudsen number
      distributions from VTK file. scripts/create_vtk.py MUST be used to generate the vtk file
    */
    vtkSmartPointer<vtkXMLStructuredGridReader> reader = vtkSmartPointer<vtkXMLStructuredGridReader>::New();
    reader->SetFileName(filePath.c_str());
    reader->Update();

    vtkSmartPointer<vtkStructuredGrid> structuredGrid = reader->GetOutput();

    vtkSmartPointer<vtkDataArray> binaryMediaArray = structuredGrid->GetPointData()->GetArray("BinaryMedia");
    vtkSmartPointer<vtkDataArray> poreSizeArray = structuredGrid->GetPointData()->GetArray("PoreSize");
    vtkSmartPointer<vtkDataArray> knudsenNumberArray = structuredGrid->GetPointData()->GetArray("KnudsenNumber");

    for (int64_t i = 0; i < dims[0]; i++)
      for (int64_t j = 0; j < dims[1]; j++)
        for (int64_t k = 0; k < dims[2]; k++)
        {
          // 2D only, 3D needs to be tested
          int index = k * dims[1] * dims[2] + i * dims[1] + j;
          binaryMedia.index_put_({i, j, k}, binaryMediaArray->GetTuple1(index));
          poreSize.index_put_({i, j, k}, poreSizeArray->GetTuple1(index));
          knudsenNumber.index_put_({i, j, k}, knudsenNumberArray->GetTuple1(index));
        }
  #endif
}

} // namespace MooseTensor
