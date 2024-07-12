//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "TensorOperatorBase.h"
#include "FFTBuffer.h"
#include "FFTProblem.h"

InputParameters
TensorOperatorBase::validParams()
{
  InputParameters params = MooseObject::validParams();
  params.registerBase("TensorOperator");
  params.addPrivateParam<FFTProblem *>("_fft_problem", nullptr);
  params.addClassDescription("TensorOperatorBase object.");
  return params;
}

TensorOperatorBase::TensorOperatorBase(const InputParameters & parameters)
  : MooseObject(parameters),
    _requested_buffers(),
    _supplied_buffers(),
    _fft_problem(*getCheckedPointerParam<FFTProblem *>("_fft_problem")),
    _x(_fft_problem.getAxis(0)),
    _y(_fft_problem.getAxis(1)),
    _z(_fft_problem.getAxis(2)),
    _i(_fft_problem.getReciprocalAxis(0)),
    _j(_fft_problem.getReciprocalAxis(1)),
    _k(_fft_problem.getReciprocalAxis(2))
{
}

const torch::Tensor &
TensorOperatorBase::getInputBuffer(const std::string & param)
{
  return getInputBufferByName(getParam<FFTInputBufferName>(param));
}

const torch::Tensor &
TensorOperatorBase::getInputBufferByName(const FFTInputBufferName & buffer_name)
{
  _requested_buffers.insert(buffer_name);
  return _fft_problem.getBuffer(buffer_name);
}

torch::Tensor &
TensorOperatorBase::getOutputBuffer(const std::string & param)
{
  return getOutputBufferByName(getParam<FFTOutputBufferName>(param));
}

torch::Tensor &
TensorOperatorBase::getOutputBufferByName(const FFTOutputBufferName & buffer_name)
{
  _supplied_buffers.insert(buffer_name);
  return _fft_problem.getBuffer(buffer_name);
}
