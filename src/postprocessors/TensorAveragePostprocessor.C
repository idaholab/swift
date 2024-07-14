//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "TensorAveragePostprocessor.h"

registerMooseObject("SwiftApp", TensorAveragePostprocessor);

InputParameters
TensorAveragePostprocessor::validParams()
{
  InputParameters params = TensorPostprocessor::validParams();
  params.addClassDescription("Compute the average value over a buffer");
  return params;
}

TensorAveragePostprocessor::TensorAveragePostprocessor(const InputParameters & parameters)
  : TensorPostprocessor(parameters)
{
}

void
TensorAveragePostprocessor::execute()
{
  _average = _u.sum().cpu().item<double>() / torch::numel(_u);
}

PostprocessorValue
TensorAveragePostprocessor::getValue() const
{
  return _average;
}
