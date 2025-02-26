/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "SR2TensorBuffer.h"

registerMooseObject("SwiftApp", SR2TensorBuffer);

InputParameters
SR2TensorBuffer::validParams()
{
  InputParameters params = TensorBufferBase::validParams();
  params.addClassDescription("Symmetric rank two tensor valued TensorBuffer object.");
  params.set<std::vector<int64_t>>("value_shape") = {6};
  params.suppressParameter<std::vector<int64_t>>("value_shape");
  return params;
}

SR2TensorBuffer::SR2TensorBuffer(const InputParameters & parameters)
  : TensorBufferBase(parameters)
{
}
