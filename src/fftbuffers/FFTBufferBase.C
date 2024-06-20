

//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTBufferBase.h"

InputParameters
FFTBufferBase::validParams()
{
  InputParameters params = MooseObject::validParams();
  params.addClassDescription("FFTBuffer object.");
  params.registerBase("FFTBuffer");
  params.registerSystemAttributeName("FFTBuffer"); //?
  params.addParam<AuxVariableName>("map_to_aux_variable",
                                   "Sync the given AuxVariable to the buffer contents");
  return params;
}

FFTBufferBase::FFTBufferBase(const InputParameters & parameters) : MooseObject(parameters) {}
