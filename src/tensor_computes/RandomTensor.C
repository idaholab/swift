/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "RandomTensor.h"
#include "SwiftUtils.h"
#include "TensorProblem.h"

registerMooseObject("SwiftApp", RandomTensor);

InputParameters
RandomTensor::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Uniform random IC with values between `min` and `max`.");
  params.addRequiredParam<Real>("min", "Minimum value.");
  params.addRequiredParam<Real>("max", "Maximum value.");
  params.addParam<int>("seed", "Random number seed.");
  return params;
}

RandomTensor::RandomTensor(const InputParameters & parameters) : TensorOperator<>(parameters) {}

void
RandomTensor::computeBuffer()
{
  const auto min = getParam<Real>("min");
  const auto max = getParam<Real>("max");
  if (isParamValid("seed"))
    torch::manual_seed(getParam<int>("seed"));
  _u = torch::rand(_tensor_problem.getShape(), MooseTensor::floatTensorOptions()) * (max - min) +
       min;
}
