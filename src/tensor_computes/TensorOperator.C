/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorOperator.h"
#include "TensorBuffer.h"
#include "TensorProblem.h"

InputParameters
TensorOperator::validParams()
{
  InputParameters params = TensorOperatorBase::validParams();
  params.addRequiredParam<TensorOutputBufferName>("buffer", "The buffer this compute is writing to");
  params.addClassDescription("TensorOperator object.");
  return params;
}

TensorOperator::TensorOperator(const InputParameters & parameters)
  : TensorOperatorBase(parameters), _u(getOutputBuffer("buffer"))
{
}
