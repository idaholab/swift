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
#include "IterativeTensorSolverInterface.h"

/**
 * BroydenSolver object
 */
class BroydenSolver : public SplitOperatorBase, public IterativeTensorSolverInterface
{
public:
  static InputParameters validParams();

  BroydenSolver(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  void broydenSolve();

  unsigned int _substep;
  unsigned int _substeps;
  unsigned int _max_iterations;

  const Real _relative_tolerance;
  const Real _absolute_tolerance;

  /// approximation of the Jacobian inverse
  torch::Tensor _M;

  const bool _verbose;
  const Real _damping;
  const Real _eye_factor;
  unsigned int _dim;
  torch::TensorOptions _options;
};
