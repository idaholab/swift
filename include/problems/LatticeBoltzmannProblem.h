/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#ifdef NEML2_ENABLED

#include "TensorProblem.h"

/**
 * Problem object for solving lattice Boltzmann problems
 */
class LatticeBoltzmannProblem : public TensorProblem
{
public:
  static InputParameters validParams();

  LatticeBoltzmannProblem(const InputParameters & parameters);

  void addStencil(const std::string & stencil_name,
                              const std::string & name,
                              InputParameters & parameters);
};

#endif
