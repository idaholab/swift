//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "ProjectTensorAux.h"
#include "SwiftTypes.h"

registerMooseObject("SwiftApp", ProjectTensorAux);

InputParameters
ProjectTensorAux::validParams()
{
  InputParameters params = AuxKernel::validParams();
  params.addClassDescription("Project an FFT buffer onto an auxiliary variable");
  params.addRequiredParam<FFTInputBufferName>("buffer", "The buffer to read from");
  return params;
}

ProjectTensorAux::ProjectTensorAux(const InputParameters & parameters)
  : AuxKernel(parameters),
    TensorProblemInterface(this),
    _cpu_buffer(_tensor_problem.getCPUBuffer(getParam<FFTInputBufferName>("buffer"))),
    _dim(_tensor_problem.getDim()),
    _n(_tensor_problem.getGridSize()),
    _grid_spacing(_tensor_problem.getGridSpacing())
{
}

Real
ProjectTensorAux::computeValue()
{
  const static Point shift(_grid_spacing[0] / 2.0, _grid_spacing[1] / 2.0, _grid_spacing[2] / 2.0);

  Point p = isNodal() ? (*_current_node + shift) : _current_elem->centroid();

  using at::indexing::TensorIndex;
  switch (_dim)
  {
    case 1:
      return _cpu_buffer.index({TensorIndex(int64_t(p(0) / _grid_spacing[0]) % _n[0])})
          .item<double>();

    case 2:
      return _cpu_buffer
          .index({TensorIndex(int64_t(p(0) / _grid_spacing[0]) % _n[0]),
                  TensorIndex(int64_t(p(1) / _grid_spacing[1]) % _n[1])})
          .item<double>();

    case 3:
      return _cpu_buffer
          .index({TensorIndex(int64_t(p(0) / _grid_spacing[0]) % _n[0]),
                  TensorIndex(int64_t(p(1) / _grid_spacing[1]) % _n[1]),
                  TensorIndex(int64_t(p(2) / _grid_spacing[2]) % _n[2])})
          .item<double>();
  }

  mooseError("Internal error (invalid dimension)");
}
