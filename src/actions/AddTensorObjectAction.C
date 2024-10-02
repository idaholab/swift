//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "AddTensorObjectAction.h"
#include "TensorProblem.h"

registerMooseAction("SwiftApp", AddTensorObjectAction, "add_tensor_ic");
registerMooseAction("SwiftApp", AddTensorObjectAction, "add_tensor_compute");
registerMooseAction("SwiftApp", AddTensorObjectAction, "add_tensor_postprocessor");

registerMooseAction("SwiftApp", AddTensorObjectAction, "add_tensor_time_integrator");

registerMooseAction("SwiftApp", AddTensorObjectAction, "add_tensor_output");

InputParameters
AddTensorObjectAction::validParams()
{
  InputParameters params = MooseObjectAction::validParams();
  params.addClassDescription("Add an TensorOperator object to the simulation.");
  return params;
}

AddTensorObjectAction::AddTensorObjectAction(const InputParameters & parameters)
  : MooseObjectAction(parameters)
{
}

void
AddTensorObjectAction::act()
{
  auto tensor_problem = std::dynamic_pointer_cast<TensorProblem>(_problem);
  if (!tensor_problem)
    mooseError("Tensor objects are only supported if the problem class is set to `TensorProblem`");

  // use addObject<Tensorxxxxxx>(_type, _name, _moose_object_pars, /* threaded = */ false) ?

  mooseInfoRepeated("AddTensorObjectAction ", _current_task, _type, _name);

  if (_current_task == "add_tensor_compute")
    tensor_problem->addTensorComputeSolve(_type, _name, _moose_object_pars);

  if (_current_task == "add_tensor_ic")
    tensor_problem->addTensorComputeInitialize(_type, _name, _moose_object_pars);

  if (_current_task == "add_tensor_postprocessor")
    tensor_problem->addTensorComputePostprocess(_type, _name, _moose_object_pars);

  if (_current_task == "add_tensor_time_integrator")
    tensor_problem->addTensorTimeIntegrator(_type, _name, _moose_object_pars);

  if (_current_task == "add_tensor_output")
    tensor_problem->addTensorOutput(_type, _name, _moose_object_pars);
}
