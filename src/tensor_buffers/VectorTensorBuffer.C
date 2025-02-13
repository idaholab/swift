/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "VectorTensorBuffer.h"

registerMooseObject("SwiftApp", VectorTensorBuffer);

InputParameters
VectorTensorBuffer::validParams()
{
  InputParameters params = TensorBufferBase::validParams();
  params.addClassDescription("Scalar valued TensorBuffer object.");
  params.set<std::vector<int64_t>>("value_shape") = {3};
  params.suppressParameter<std::vector<int64_t>>("value_shape");
  return params;
}

VectorTensorBuffer::VectorTensorBuffer(const InputParameters & parameters)
  : TensorBufferBase(parameters)
{
}
