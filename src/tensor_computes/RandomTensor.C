//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "RandomTensor.h"
#include "SwiftUtils.h"
#include "TensorProblem.h"

registerMooseObject("SwiftApp", RandomTensor);

InputParameters
RandomTensor::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription("Uniform random IC with values between `min` and `max`.");
  params.addRequiredParam<Real>("min", "Minimum value.");
  params.addRequiredParam<Real>("max", "Maximum value.");
  return params;
}

RandomTensor::RandomTensor(const InputParameters & parameters) : TensorOperator(parameters)
{
}

void
RandomTensor::computeBuffer()
{
  const auto min = getParam<Real>("min");
  const auto max = getParam<Real>("max");
  _u = torch::rand(_tensor_problem.getShape(), MooseTensor::floatTensorOptions()) * (max - min) +
       min;
}
