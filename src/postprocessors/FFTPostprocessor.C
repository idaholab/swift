//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTProblem.h"
#include "DependencyResolverInterface.h"

registerMooseObject("SwiftApp", FFTProblem);

InputParameters
FFTPostprocessor::validParams()
{
  InputParameters params = GeneralPostprocessor::validParams();
  params.addClassDescription("A normal Postprocessor acting on ann FFT buffer.");
  params.addRequiredParam<FFTInputBufferName>("buffer", "The buffer this compute is operating on");
  return params;
}

FFTPostprocessor::FFTPostprocessor(const InputParameters & parameters)
  : GeneralPostprocessor(parameters),
    _fft_problem(*parameters.getCheckedPointerParam<FFTProblem *>(
        "_fe_problem", "FFTPostprocessors require a FFTProblem.")),
    _u(_fft_problem.getBuffer(getParam<FFTInputBufferName>("buffer")))
{
}
