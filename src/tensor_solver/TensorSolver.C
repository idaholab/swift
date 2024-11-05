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
#include "SwiftTypes.h"

InputParameters
TensorSolver::validParams()
{
  InputParameters params = TensorOperatorBase::validParams();
  params.registerBase("TensorSolver");
  params.addParam<TensorComputeName>(
      "root_compute",
      "Primary compute object that updates the buffers. This is usually a "
      "ComputeGroup object. A ComputeGroup encompassing all computes will be generated "
      "automatically if the user does not provide this parameter.");
  params.addClassDescription("TensorSolver object.");
  return params;
}

TensorSolver::TensorSolver(const InputParameters & parameters)
  : TensorOperatorBase(parameters), _dt(_tensor_problem.dt()), _dt_old(_tensor_problem.dtOld())
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

void
TensorSolver::updateDependencies()
{
  // the compute that's being solved for (usually a ComputeGroup)
  const auto & root_name = getParam<TensorComputeName>("root_compute");
  for (const auto cmp : _tensor_problem.getComputes())
    if (cmp->name() == root_name)
    {
      _compute = cmp;
      _compute->updateDependencies();
      return;
    }

  paramError("root_compute", "Compute object not found.");
}
