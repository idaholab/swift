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
    _fft_problem(dynamic_cast<FFTProblem *>(&_subproblem)),
    _buffer(
        [this]()
        {
          if (!_fft_problem)
            mooseError("Can only be used with FFTProblem");
          return _fft_problem->getBuffer(getParam<FFTInputBufferName>("buffer"));
        }())
{
}

Real
FFTBufferAux::computeValue()
{
  if (isNodal())
  {
    Point p = *_current_node;
    return 0.0;
  }
  else
  {
    Point p = _current_elem->centroid();
    return 1.0;
  }
}
