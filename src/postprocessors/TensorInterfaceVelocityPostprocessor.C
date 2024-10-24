//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "TensorInterfaceVelocityPostprocessor.h"
#include "DomainAction.h"
#include "TensorProblem.h"

registerMooseObject("SwiftApp", TensorInterfaceVelocityPostprocessor);

InputParameters
TensorInterfaceVelocityPostprocessor::validParams()
{
  InputParameters params = TensorPostprocessor::validParams();
  params.addClassDescription("Compute the integral over a buffer");
  return params;
}

TensorInterfaceVelocityPostprocessor::TensorInterfaceVelocityPostprocessor(
    const InputParameters & parameters)
  : TensorPostprocessor(parameters),
    _u_old(_tensor_problem.getBufferOld(getParam<TensorInputBufferName>("buffer"), 1)),
    _dim(_domain.getDim()),
    _i(torch::tensor(c10::complex<double>(0.0, 1.0), MooseTensor::complexFloatTensorOptions()))
{
}

void
TensorInterfaceVelocityPostprocessor::execute()
{
  if (_u_old.empty())
  {
    _velocity = 0.0;
    return;
  }

  const auto du = (_u - _u_old[0]) / _dt; // TODO: _dt_old?
  torch::Tensor vsquare;
  for (const auto i : make_range(_dim))
  {
    const auto grad = _domain.ifft(_domain.fft(_u) * _domain.getReciprocalAxis(i) * _i);
    const auto v = torch::where(torch::abs(grad) > 1e-3, du / grad, 0.0);
    if (i == 0)
      vsquare = v * v;
    else
      vsquare += v * v;
  }

  _velocity = std::sqrt(torch::max(vsquare).item<double>());
}

PostprocessorValue
TensorInterfaceVelocityPostprocessor::getValue() const
{
  return _velocity;
}
