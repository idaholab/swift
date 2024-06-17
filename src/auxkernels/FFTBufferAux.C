//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTBufferAux.h"
#include "FFTProblem.h"
#include "SwiftTypes.h"

registerMooseObject("SwiftApp", FFTBufferAux);

InputParameters
FFTBufferAux::validParams()
{
  InputParameters params = AuxKernel::validParams();
  params.addClassDescription("Project an FFT buffer onto an auxiliary variable");
  params.addRequiredParam<FFTInputBufferName>("buffer", "The buffer to read from");
  return params;
}

FFTBufferAux::FFTBufferAux(const InputParameters & parameters)
  : AuxKernel(parameters),
    _fft_problem(
        [this]()
        {
          auto fft_problem = dynamic_cast<FFTProblem *>(&_subproblem);
          if (!fft_problem)
            mooseError("Can only be used with FFTProblem");
          return fft_problem;
        }()),
    _buffer(_fft_problem->getBuffer(getParam<FFTInputBufferName>("buffer"))),
    _dim(_fft_problem->getDim()),
    _n(_fft_problem->getGridSize()),
    _grid_spacing(_fft_problem->getGridSpacing())
{
}

void
FFTBufferAux::customSetup(const ExecFlagType & e)
{
  if (!_execute_enum.contains(e))
    return;
  _cpu_buffer = _buffer.cpu();
}

Real
FFTBufferAux::computeValue()
{
  const static Point shift(_grid_spacing[0] / 2.0, _grid_spacing[1] / 2.0, _grid_spacing[2] / 2.0);

  Point p = isNodal() ? (*_current_node + shift) : _current_elem->centroid();

  switch (_dim)
  {
    case 1:
      return _cpu_buffer.index({static_cast<long int>(p(0) / _grid_spacing[0]) % _n[0]})
          .item<double>();

    case 2:
      return _cpu_buffer
          .index({static_cast<long int>(p(0) / _grid_spacing[0]) % _n[0],
                  static_cast<long int>(p(1) / _grid_spacing[1]) % _n[1]})
          .item<double>();

    case 3:
      return _cpu_buffer
          .index({static_cast<long int>(p(0) / _grid_spacing[0]) % _n[0],
                  static_cast<long int>(p(1) / _grid_spacing[1]) % _n[1],
                  static_cast<long int>(p(2) / _grid_spacing[2]) % _n[2]})
          .item<double>();
  }

  mooseError("Internal error (invalid dimension)");
}
