//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "TensorSolver.h"
#include "TensorProblem.h"

InputParameters
TensorSolver::validParams()
{
  InputParameters params = TensorOperatorBase::validParams();
  params.registerBase("TensorSolver");
  params.addClassDescription("TensorSolver object.");
  return params;
}

TensorSolver::TensorSolver(const InputParameters & parameters)
  : TensorOperatorBase(parameters),
    _computes(_tensor_problem.getComputes()),
    _dt(_tensor_problem.dt()),
    _dt_old(_tensor_problem.dtOld())
{
}

const std::vector<torch::Tensor> &
TensorSolver::getBufferOld(const std::string & param, unsigned int max_states)
{
  return getBufferOldByName(getParam<TensorInputBufferName>(param), max_states);
}

const std::vector<torch::Tensor> &
TensorSolver::getBufferOldByName(const TensorInputBufferName & buffer_name, unsigned int max_states)
{
  return _tensor_problem.getBufferOld(buffer_name, max_states);
}
