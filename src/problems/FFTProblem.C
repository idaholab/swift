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

  _xmax = fft_mesh->getMaxInDimension(0);
  _ymax = fft_mesh->getMaxInDimension(1);
  _zmax = fft_mesh->getMaxInDimension(2);
  _nx = fft_mesh->getElementsInDimension(0);
  _ny = fft_mesh->getElementsInDimension(1);
  _nz = fft_mesh->getElementsInDimension(2);

  switch (_dim)
  {
    case 1:
      _shape_storage = {_nx};
    case 2:
      _shape_storage = {_nx, _ny};
    case 3:
      _shape_storage = {_nx, _ny, _nz};
    default:
      mooseError("Unsupported mesh dimension");
  }
  _shape = _shape_storage;

  // Compute grid spacing
  _dx = _xmax / _nx;
  _dy = _ymax / _ny;
  _dz = _zmax / _nz;

  // initialize tensors (assuming all scalar for now, but in the future well have an FFTBufferBase
  // pointer as well)
  for (auto pair : _fft_buffer)
    pair.second = torch::zeros(_shape);
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
