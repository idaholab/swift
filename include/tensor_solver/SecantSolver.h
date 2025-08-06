/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "SplitOperatorBase.h"
#include "IterativeTensorSolverInterface.h"

/**
 * SecantSolver object
 */
class SecantSolver : public SplitOperatorBase, public IterativeTensorSolverInterface
{
public:
  static InputParameters validParams();

  SecantSolver(const InputParameters & parameters);
  ~SecantSolver();

  virtual void computeBuffer() override;

protected:
  void secantSolve();

  unsigned int _substep;
  unsigned int _substeps;
  unsigned int _max_iterations;

  const Real _relative_tolerance;
  const Real _absolute_tolerance;

  const bool _verbose;
  const Real _damping;
  const Real _trust_radius;

  const bool _adaptive_damping;
  const Real _adaptive_damping_cutback_factor;
  const Real _adaptive_damping_growth_factor;

  const Real _dt_epsilon;

  unsigned int _total_iterations;
};
