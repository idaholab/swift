//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "ScalarTensorBuffer.h"

registerMooseObject("SwiftApp", ScalarTensorBuffer);

InputParameters
ScalarTensorBuffer::validParams()
{
  InputParameters params = TensorBufferBase::validParams();
  params.addClassDescription("Add a scalar valued TensorBuffer object.");
  return params;
}

ScalarTensorBuffer::ScalarTensorBuffer(const InputParameters & parameters) : TensorBufferBase(parameters) {}
