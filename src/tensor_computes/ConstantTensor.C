//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "ConstantTensor.h"
#include "SwiftUtils.h"

registerMooseObject("SwiftApp", ConstantTensor);

InputParameters
ConstantTensor::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription("Constant tensor value.");
  params.addRequiredParam<Real>("real", "Real part of the constant value.");
  params.addParam<Real>(
      "imaginary", 0.0, "Imaginary part of the constant value (only for reciprocal buffers).");
  params.addParam<bool>("reciprocal", false, "Construct a reciprocal buffer");
  params.addParam<bool>("full", false, "Construct a full tensor will all entries");
  return params;
}

ConstantTensor::ConstantTensor(const InputParameters & parameters) : TensorOperator(parameters) {}

void
ConstantTensor::computeBuffer()
{
  const auto scalar = getParam<Real>("real");
  _u = torch::tensor({scalar}, MooseTensor::floatTensorOptions());
}
