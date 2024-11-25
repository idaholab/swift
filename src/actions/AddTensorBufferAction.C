/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

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
