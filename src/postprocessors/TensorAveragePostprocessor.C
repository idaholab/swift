//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTAveragePostprocessor.h"

registerMooseObject("SwiftApp", FFTAveragePostprocessor);

InputParameters
FFTAveragePostprocessor::validParams()
{
  InputParameters params = FFTPostprocessor::validParams();
  params.addClassDescription("Compute the average value over a buffer");
  return params;
}

FFTAveragePostprocessor::FFTAveragePostprocessor(const InputParameters & parameters)
  : FFTPostprocessor(parameters)
{
}

void
FFTAveragePostprocessor::execute()
{
  _average = _u.sum().cpu().item<double>() / torch::numel(_u);
}

PostprocessorValue
FFTAveragePostprocessor::getValue() const
{
  return _average;
}
