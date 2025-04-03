/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorSolver.h"

/**
 * TensorTimeIntegrator object (this is mostly a compute object)
 */
class ExplicitSolverBase : public TensorSolver
{
public:
  static InputParameters validParams();

  ExplicitSolverBase(const InputParameters & parameters);

protected:
  struct Variable
  {
    torch::Tensor & _buffer;
    const torch::Tensor & _reciprocal_buffer;
    const torch::Tensor & _time_derivative_reciprocal;
  };

  std::vector<Variable> _variables;
};
