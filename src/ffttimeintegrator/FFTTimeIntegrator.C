//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTTimeIntegrator.h"
#include "FFTProblem.h"

InputParameters
FFTTimeIntegrator::validParams()
{
  InputParameters params = FFTCompute::validParams();
  params.registerBase("FFTTimeIntegrator");
  params.addClassDescription("FFTTimeIntegrator object.");
  return params;
}

FFTTimeIntegrator::FFTTimeIntegrator(const InputParameters & parameters)
  : FFTCompute(parameters), _dt(_fft_problem.dt())
{
}

const std::vector<torch::Tensor> &
FFTTimeIntegrator::getBufferOld(const std::string & param, unsigned int max_states)
{
  return getBufferOldByName(getParam<FFTInputBufferName>(param), max_states);
}

const std::vector<torch::Tensor> &
FFTTimeIntegrator::getBufferOldByName(const FFTInputBufferName & buffer_name,
                                      unsigned int max_states)
{
  return _fft_problem.getBufferOld(buffer_name, max_states);
}
