//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "RandomFFTIC.h"
#include "SwiftUtils.h"
#include "FFTProblem.h"

registerMooseObject("SwiftApp", RandomFFTIC);

InputParameters
RandomFFTIC::validParams()
{
  InputParameters params = FFTInitialCondition::validParams();
  params.addClassDescription("Uniform random IC with values between `min` and `max`.");
  params.addRequiredParam<Real>("min", "Minimum value.");
  params.addRequiredParam<Real>("max", "Maximum value.");
  return params;
}

RandomFFTIC::RandomFFTIC(const InputParameters & parameters) : FFTInitialCondition(parameters) {}

void
RandomFFTIC::computeBuffer()
{
  const auto min = getParam<Real>("min");
  const auto max = getParam<Real>("max");
  _u = torch::rand(_fft_problem.getShape(), MooseFFT::floatTensorOptions()) * (max - min) + min;
}
