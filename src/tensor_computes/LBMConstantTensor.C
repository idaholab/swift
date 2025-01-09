/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMConstantTensor.h"

registerMooseObject("SwiftApp", LBMConstantTensor);

InputParameters
LBMConstantTensor::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addParam<Real>("value", 0.0, "The constant value.");
  params.addClassDescription("LBMConstantTensor object.");
  return params;
}

LBMConstantTensor::LBMConstantTensor(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters)
{
}

void
LBMConstantTensor::computeBuffer()
{
  const auto value = getParam<Real>("value");
  _u.fill_(value);
}
