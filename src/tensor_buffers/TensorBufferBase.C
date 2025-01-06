/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorBufferBase.h"

InputParameters
TensorBufferBase::validParams()
{
  InputParameters params = MooseObject::validParams();
  params.addClassDescription("TensorBuffer object.");
  params.registerBase("TensorBuffer");
  params.registerSystemAttributeName("TensorBuffer"); //?
  params.addParam<AuxVariableName>("map_to_aux_variable",
                                   "Sync the given AuxVariable to the buffer contents");
  params.addParam<unsigned int>("vector_size", 1, "Add extra dimension to tensors.");
  return params;
}

TensorBufferBase::TensorBufferBase(const InputParameters & parameters) : MooseObject(parameters) {}
