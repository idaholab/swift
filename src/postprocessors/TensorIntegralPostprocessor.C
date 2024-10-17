//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "TensorIntegralPostprocessor.h"
#include "DomainAction.h"
#include "TensorProblem.h"
#include "UniformTensorMesh.h"

registerMooseObject("SwiftApp", TensorIntegralPostprocessor);

InputParameters
TensorIntegralPostprocessor::validParams()
{
  InputParameters params = TensorPostprocessor::validParams();
  params.addClassDescription("Compute the integral over a buffer");
  return params;
}

TensorIntegralPostprocessor::TensorIntegralPostprocessor(const InputParameters & parameters)
  : TensorPostprocessor(parameters)
{
}

void
TensorIntegralPostprocessor::execute()
{
  _integral = _u.sum().cpu().item<double>();

  for (const auto dim : make_range(_domain.getDim()))
    _integral *= _domain.getDomainSize()[dim];

  _integral /= _u.numel();
}

PostprocessorValue
TensorIntegralPostprocessor::getValue() const
{
  return _integral;
}
