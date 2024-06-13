//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "ScalarFFTBuffer.h"

registerMooseObject("SwiftApp", ScalarFFTBuffer);

InputParameters
ScalarFFTBuffer::validParams()
{
  InputParameters params = FFTBufferBase::validParams();
  params.addClassDescription("Add a scalar valued FFTBuffer object.");
  return params;
}

ScalarFFTBuffer::ScalarFFTBuffer(const InputParameters & parameters) : FFTBufferBase(parameters) {}
