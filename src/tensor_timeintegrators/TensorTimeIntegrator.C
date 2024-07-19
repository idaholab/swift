//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "TensorTimeIntegrator.h"
#include "TensorProblem.h"

InputParameters
TensorTimeIntegrator::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.registerBase("TensorTimeIntegrator");
  params.addClassDescription("TensorTimeIntegrator object.");
  return params;
}

TensorTimeIntegrator::TensorTimeIntegrator(const InputParameters & parameters)
  : TensorOperator(parameters), _dt(_tensor_problem.getSubDt())
{
}

const std::vector<torch::Tensor> &
TensorTimeIntegrator::getBufferOld(const std::string & param, unsigned int max_states)
{
  return getBufferOldByName(getParam<FFTInputBufferName>(param), max_states);
}

const std::vector<torch::Tensor> &
TensorTimeIntegrator::getBufferOldByName(const FFTInputBufferName & buffer_name,
                                      unsigned int max_states)
{
  return _tensor_problem.getBufferOld(buffer_name, max_states);
}
