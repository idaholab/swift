//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "SineIC.h"

registerMooseObject("SwiftApp", SineIC);

InputParameters
SineIC::validParams()
{
  InputParameters params = FFTInitialCondition::validParams();
  params.addClassDescription("Sinusoidal IC.");
  return params;
}

SineIC::SineIC(const InputParameters & parameters) : FFTInitialCondition(parameters) {}

void
SineIC::computeBuffer()
{
  _u = torch::sin(_x) + torch::sin(_y) + torch::sin(_z);
}
