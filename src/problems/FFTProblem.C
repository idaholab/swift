//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTProblem.h"
#include "FFTMesh.h"
#include "SwiftUtils.h"

registerMooseObject("SwiftApp", FFTProblem);

InputParameters
FFTProblem::validParams()
{
  InputParameters params = FEProblem::validParams();
  params.addClassDescription(
      "A normal Problem object that adds the ability to perform spectral solves.");
  params.set<bool>("skip_nl_system_check") = true;
  return params;
}

FFTProblem::FFTProblem(const InputParameters & parameters) : FEProblem(parameters) {}

void
FFTProblem::initialSetup()
{
  auto * fft_mesh = dynamic_cast<FFTMesh *>(&_mesh);
  if (!fft_mesh)
    mooseError("FFTProblem must be used with an FFTMesh");

  _dim = fft_mesh->getDim();

  // get grid geometry
  for (const auto dim : make_range(3))
  {
    _max[dim] = fft_mesh->getMaxInDimension(dim);
    _n[dim] = fft_mesh->getElementsInDimension(dim);
    _grid_spacing[dim] = _max[dim] / _n[dim];
  }

  switch (_dim)
  {
    case 1:
      _shape_storage = {_n[0]};
    case 2:
      _shape_storage = {_n[0], _n[1]};
    case 3:
      _shape_storage = {_n[0], _n[1], _n[2]};
    default:
      mooseError("Unsupported mesh dimension");
  }
  _shape = _shape_storage;

  // initialize tensors (assuming all scalar for now, but in the future well have an FFTBufferBase
  // pointer as well)
  auto options = MooseFFT::floatTensorOptions();
  for (auto pair : _fft_buffer)
    pair.second = torch::zeros(_shape, options);

  // build real space axes
  for (const auto dim : make_range(_dim))
    _axis.push_back(
        torch::unsqueeze(torch::linspace(c10::Scalar(_grid_spacing[dim] / 2.0),
                                         c10::Scalar(_max[dim] - _grid_spacing[dim] / 2.0),
                                         _n[dim],
                                         options),
                         _dim - dim - 1));

  // build reciprocal space axes
  for (const auto dim : make_range(_dim))
  {
    const auto freq = (dim == _dim - 1) ? torch::fft::rfftfreq(_n[dim], _grid_spacing[dim], options)
                                        : torch::fft::fftfreq(_n[dim], _grid_spacing[dim], options);
    _reciprocal_axis.push_back(torch::unsqueeze(freq, _dim - dim - 1));
  }

  // dependency resolution of FFTICs

  // dependency resolution of FFTComputes
}

void
FFTProblem::addFFTBuffer(const std::string & buffer_name, InputParameters & parameters)
{
  if (_fft_buffer.find(buffer_name) != _fft_buffer.end())
    mooseError("FFTBuffer '", buffer_name, "' already exists in the system");
  _fft_buffer.try_emplace(buffer_name);
}

void
FFTProblem::addFFTCompute(const std::string & compute_name, InputParameters & parameters)
{
}
