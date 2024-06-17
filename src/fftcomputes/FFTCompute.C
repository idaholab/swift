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
  params.addRequiredParam<FFTOutputBufferName>("buffer", "The buffer this compute is writing to");
  params.registerBase("FFTCompute");
  params.addClassDescription("FFTCompute object.");
  return params;
}

FFTCompute::FFTCompute(const InputParameters & parameters)
  : MooseObject(parameters),
    _fft_problem(*parameters.getCheckedPointerParam<FFTProblem *>("_fft_problem")),
    _u(getOutputBuffer("buffer")),
    _x(_fft_problem.getAxis(0)),
    _y(_fft_problem.getAxis(1)),
    _z(_fft_problem.getAxis(2)),
    _requested_buffers(),
    _supplied_buffers()
{
  mooseInfoRepeated("ctor _requested_buffers ", _requested_buffers.size());
}

const torch::Tensor &
FFTCompute::getInputBuffer(const std::string & param)
{
  return getInputBufferByName(getParam<FFTInputBufferName>(param));
}

const torch::Tensor &
FFTCompute::getInputBufferByName(const FFTInputBufferName & buffer_name)
{
  _supplied_buffers.insert(buffer_name);
  return _fft_problem.getBuffer(buffer_name);
}

torch::Tensor &
FFTCompute::getOutputBuffer(const std::string & param)
{
  mooseInfoRepeated("getOutputBuffer _requested_buffers ", _requested_buffers.size());
  return getOutputBufferByName(getParam<FFTOutputBufferName>(param));
}

torch::Tensor &
FFTCompute::getOutputBufferByName(const FFTOutputBufferName & buffer_name)
{
  mooseInfoRepeated(_requested_buffers.size(), buffer_name);
  _requested_buffers.insert(buffer_name);
  return _fft_problem.getBuffer(buffer_name);
}
