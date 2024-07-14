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
#include "TensorProblem.h"

InputParameters
TensorOperatorBase::validParams()
{
  InputParameters params = MooseObject::validParams();
  params.registerBase("TensorOperator");
  params.addPrivateParam<TensorProblem *>("_tensor_problem", nullptr);
  params.addClassDescription("TensorOperatorBase object.");
  return params;
}

TensorOperatorBase::TensorOperatorBase(const InputParameters & parameters)
  : MooseObject(parameters),
    _requested_buffers(),
    _supplied_buffers(),
    _tensor_problem(*getCheckedPointerParam<TensorProblem *>("_tensor_problem")),
    _x(_tensor_problem.getAxis(0)),
    _y(_tensor_problem.getAxis(1)),
    _z(_tensor_problem.getAxis(2)),
    _i(_tensor_problem.getReciprocalAxis(0)),
    _j(_tensor_problem.getReciprocalAxis(1)),
    _k(_tensor_problem.getReciprocalAxis(2))
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
  return _tensor_problem.getBuffer(buffer_name);
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
  return _tensor_problem.getBuffer(buffer_name);
}
