
//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "PerformFFT.h"

registerMooseObject("SwiftApp", PerformFFT);

InputParameters
PerformFFT::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription("PerformFFT object.");
  params.addParam<FFTInputBufferName>("input", "Input buffer name");
  params.addParam<bool>("forward", true, "Forward FFT");
  return params;
}

PerformFFT::PerformFFT(const InputParameters & parameters)
  : TensorOperator(parameters), _forward(getParam<bool>("forward")), _input(getInputBuffer("input"))
{
}

void
PerformFFT::computeBuffer()
{
  if (_forward)
    _u = _tensor_problem.fft(_input);
  else
    _u = _tensor_problem.ifft(_input);
}
