//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "SplitOperatorBase.h"

/**
 * SecantSolver object
 */
class SecantSolver : public SplitOperatorBase
{
public:
  static InputParameters validParams();

  SecantSolver(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  void secantSolve();

  unsigned int _substep;
  unsigned int _substeps;
  unsigned int _max_iterations;
  const Real _tolerance;
  const bool _verbose;
};
