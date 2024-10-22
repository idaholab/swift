
//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "CreateTensorSolverAction.h"
#include "TensorProblem.h"
#include "TensorSolver.h"

registerMooseAction("SwiftApp", CreateTensorSolverAction, "create_tensor_solver");

InputParameters
CreateTensorSolverAction::validParams()
{
  InputParameters params = MooseObjectAction::validParams();
  params.addClassDescription("Create a TensorSolver.");
  return params;
}

CreateTensorSolverAction::CreateTensorSolverAction(const InputParameters & parameters)
  : MooseObjectAction(parameters), DomainInterface(this)
{
}

void
CreateTensorSolverAction::act()
{
  auto tensor_problem = std::dynamic_pointer_cast<TensorProblem>(_problem);
  if (!tensor_problem)
    mooseError("A TensorSolver is only supported if the problem class is set to `TensorProblem`");

  // Add a pointer to the TensorProblem and the Domain
  _moose_object_pars.addPrivateParam<TensorProblem *>("_tensor_problem", tensor_problem.get());
  _moose_object_pars.addPrivateParam<const DomainAction *>("_domain", &_domain);

  // Create the object
  auto solver = _factory.create<TensorSolver>(_type, "TensorSolver", _moose_object_pars, 0);
  tensor_problem->setSolver(solver, {});
}
