//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTExtremeValuePostprocessor.h"

registerMooseObject("SwiftApp", FFTExtremeValuePostprocessor);

InputParameters
FFTExtremeValuePostprocessor::validParams()
{
  InputParameters params = FFTPostprocessor::validParams();
  params.addClassDescription("Find extreme values in the FFT buffer");
  MooseEnum valueType("MIN MAX");
  params.addParam<MooseEnum>("value_type", valueType, "Extreme value type");
  return params;
}

FFTExtremeValuePostprocessor::FFTExtremeValuePostprocessor(const InputParameters & parameters)
  : FFTPostprocessor(parameters),
    _value_type(getParam<MooseEnum>("value_type").getEnum<ValueType>())
{
}

void
FFTExtremeValuePostprocessor::execute()
{
  _value = _value_type == ValueType::MIN ? torch::min(_u).cpu().item<double>()
                                         : torch::max(_u).cpu().item<double>();
}

PostprocessorValue
FFTExtremeValuePostprocessor::getValue() const
{
  return _value;
}
