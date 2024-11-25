/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ScalarTensorBuffer.h"

registerMooseObject("SwiftApp", ScalarTensorBuffer);

InputParameters
ScalarTensorBuffer::validParams()
{
  InputParameters params = TensorBufferBase::validParams();
  params.addClassDescription("Add a scalar valued TensorBuffer object.");
  return params;
}

ScalarTensorBuffer::ScalarTensorBuffer(const InputParameters & parameters) : TensorBufferBase(parameters) {}
