//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "TensorOperator.h"
#include "TensorBuffer.h"
#include "TensorProblem.h"

InputParameters
TensorOperator::validParams()
{
  InputParameters params = TensorOperatorBase::validParams();
  params.addRequiredParam<TensorOutputBufferName>("buffer", "The buffer this compute is writing to");
  params.addClassDescription("TensorOperator object.");
  return params;
}

TensorOperator::TensorOperator(const InputParameters & parameters)
  : TensorOperatorBase(parameters), _u(getOutputBuffer("buffer"))
{
}
