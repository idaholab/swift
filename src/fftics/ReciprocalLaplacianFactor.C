//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "ReciprocalLaplacianFactor.h"

registerMooseObject("SwiftApp", ReciprocalLaplacianFactor);

InputParameters
ReciprocalLaplacianFactor::validParams()
{
  InputParameters params = FFTInitialCondition::validParams();
  params.addClassDescription("Reciprocal space Laplacian IC.");
  params.addParam<Real>("factor", 1.0, "Prefactor");
  return params;
}

ReciprocalLaplacianFactor::ReciprocalLaplacianFactor(const InputParameters & parameters)
  : FFTInitialCondition(parameters), _factor(getParam<Real>("factor"))
{
}

void
ReciprocalLaplacianFactor::computeBuffer()
{
  _u = (_i * _i + _j * _j + _k * _k) * (libMesh::pi * libMesh::pi * _factor);
}
