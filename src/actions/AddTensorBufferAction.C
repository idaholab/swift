//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "AddTensorBufferAction.h"
#include "TensorProblem.h"

registerMooseAction("SwiftApp", AddTensorBufferAction, "add_tensor_buffer");

InputParameters
AddTensorBufferAction::validParams()
{
  InputParameters params = MooseObjectAction::validParams();
  params.addClassDescription("Add an TensorBuffer object to the simulation.");
  params.set<std::string>("type") = "ScalarTensorBuffer";
  return params;
}

AddTensorBufferAction::AddTensorBufferAction(const InputParameters & parameters)
  : MooseObjectAction(parameters)
{
}

void
AddTensorBufferAction::act()
{
  auto tensor_problem = std::dynamic_pointer_cast<TensorProblem>(_problem);
  if (!tensor_problem)
    mooseError("Tensor Buffers are only supported if the problem class is set to `TensorProblem`");

  tensor_problem->addTensorBuffer(_name, _moose_object_pars);
}
