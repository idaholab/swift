/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2025 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "ExplicitSolverBase.h"

/**
 * Runge–Kutta–Chebyshev (RKC) explicit stabilized time integrator.
 *
 * This implements the explicit, multi-stage stabilized Runge–Kutta scheme based on
 * Chebyshev polynomials. The implementation mirrors the style of other tensor solvers.
 *
 * Notes:
 * - The single-stage case reduces to Forward Euler.
 * - Multi-stage stabilized variants (s > 1) require specific coefficients; a follow-up
 *   patch can extend this to classic RKC2/ROCK2 after user confirmation.
 */
class RungeKuttaChebyshev : public ExplicitSolverBase
{
public:
  static InputParameters validParams();

  RungeKuttaChebyshev(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  unsigned int _substeps;
  unsigned int _stages;     ///< number of RKC stages (>=1)
  Real _damping;            ///< damping parameter epsilon (>=0)

  Real & _sub_dt;
  Real & _sub_time;
};

