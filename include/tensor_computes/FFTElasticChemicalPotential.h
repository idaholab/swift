//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "TensorOperator.h"

/**
 * Chemical potential for small strain elasticity volumetric Eigenstrain solute
 */
class FFTElasticChemicalPotential : public TensorOperator
{
public:
  static InputParameters validParams();

  FFTElasticChemicalPotential(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  std::vector<const torch::Tensor *> _displacements;
  const torch::Tensor _two_pi_i;
  const Real _mu;
  const Real _lambda;
  const Real _e0;
  const torch::Tensor & _cbar;
};
