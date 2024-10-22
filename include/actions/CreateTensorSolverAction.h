
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
#include "DomainInterface.h"

/**
 * This class creates the TensorSolver object.
 */
class CreateTensorSolverAction : public MooseObjectAction, public DomainInterface
{
public:
  static InputParameters validParams();

  CreateTensorSolverAction(const InputParameters & parameters);

  virtual void act() override;
};
