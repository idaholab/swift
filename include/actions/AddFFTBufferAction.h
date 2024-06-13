//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "MooseObjectAction.h"

class FFTProblem;

/**
 * This class adds an FFTBuffer object.
 * The FFTBuffer is a structured grid object using a libtorch tensor to store data
 * in real space. A reciprocal space representation is automatically created on demand.
 */
class AddFFTBufferAction : public MooseObjectAction
{
public:
  static InputParameters validParams();

  AddFFTBufferAction(const InputParameters & parameters);

  virtual void act() override;
};
