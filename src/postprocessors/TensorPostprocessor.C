//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "TensorPostprocessor.h"
#include "TensorProblem.h"

registerMooseObject("SwiftApp", TensorProblem);

InputParameters
TensorPostprocessor::validParams()
{
  InputParameters params = GeneralPostprocessor::validParams();
  params.addClassDescription("A normal Postprocessor acting on an FFT buffer.");
  params.addRequiredParam<FFTInputBufferName>("buffer", "The buffer this compute is operating on");
  return params;
}

TensorPostprocessor::TensorPostprocessor(const InputParameters & parameters)
  : GeneralPostprocessor(parameters),
    _tensor_problem(
        [this]()
        {
          auto tensor_problem = dynamic_cast<TensorProblem *>(&_fe_problem);
          if (!tensor_problem)
            mooseError("FFTPostprocessors require a TensorProblem.");
          return std::ref(*tensor_problem);
        }()),
    _u(_tensor_problem.getBuffer(getParam<FFTInputBufferName>("buffer")))
{
}
