//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTCompute.h"
#include "FFTBuffer.h"
#include "FFTProblem.h"

InputParameters
FFTCompute::validParams()
{
  InputParameters params = MooseObject::validParams();
  params.addRequiredParam<FFTBufferName>("output", "The buffer this compute is writing to");
  params.registerBase("FFTCompute");
  params.addClassDescription("FFTCompute object.");
  return params;
}

FFTCompute::FFTCompute(const InputParameters & parameters)
  : MooseObject(parameters),
    _u(getBuffer("output")),
    _fft_problem(*parameters.getCheckedPointerParam<FFTProblem *>("_fft_problem"))
{
}

torch::Tensor &
FFTCompute::getBuffer(const std::string & param)
{
  return getBufferByName(getParam<FFTBufferName>(param));
}

torch::Tensor &
FFTCompute::getBufferByName(const FFTBufferName & buffer_name)
{
  return _fft_problem.getBuffer(buffer_name);
}
