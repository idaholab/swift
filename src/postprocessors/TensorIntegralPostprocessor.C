/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorIntegralPostprocessor.h"
#include "DomainAction.h"
#include "TensorProblem.h"

registerMooseObject("SwiftApp", TensorIntegralPostprocessor);

InputParameters
TensorIntegralPostprocessor::validParams()
{
  InputParameters params = TensorPostprocessor::validParams();
  params.addClassDescription("Compute the integral over a buffer");
  return params;
}

TensorIntegralPostprocessor::TensorIntegralPostprocessor(const InputParameters & parameters)
  : TensorPostprocessor(parameters)
{
}

void
TensorIntegralPostprocessor::execute()
{
  _integral = _u.sum().cpu().item<double>();

  for (const auto dim : make_range(_domain.getDim()))
    _integral *= _domain.getDomainMax()[dim] - _domain.getDomainMin()[dim];

  _integral /= _u.numel();
}

PostprocessorValue
TensorIntegralPostprocessor::getValue() const
{
  return _integral;
}
