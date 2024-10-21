//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "TensorIntegralChangePostprocessor.h"
#include "DomainAction.h"
#include "TensorProblem.h"
#include "UniformTensorMesh.h"

registerMooseObject("SwiftApp", TensorIntegralChangePostprocessor);

InputParameters
TensorIntegralChangePostprocessor::validParams()
{
  InputParameters params = TensorPostprocessor::validParams();
  params.addClassDescription("Compute the integral over a buffer");
  return params;
}

TensorIntegralChangePostprocessor::TensorIntegralChangePostprocessor(const InputParameters & parameters)
  : TensorPostprocessor(parameters), _u_old(_tensor_problem.getBufferOld(getParam<TensorInputBufferName>("buffer"), 1))
{
}

void
TensorIntegralChangePostprocessor::execute()
{
  {
  if (!_u_old.empty())
    _integral = torch::abs(_u - _u_old[0]).sum().cpu().item<double>();
  else
    _integral = torch::abs(_u).sum().cpu().item<double>();

  for (const auto dim : make_range(_domain.getDim()))
    _integral *= _domain.getGridSpacing()[dim];
  }
}

PostprocessorValue
TensorIntegralChangePostprocessor::getValue() const
{
  return _integral;
}
