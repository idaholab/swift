//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "DirichletBC.h"

/**
 * Boundary condition of a Dirichlet type
 *
 * Sets the value in the node
 */
class CoordinatePlaneNodalContactBC : public DirichletBC
{
public:
  static InputParameters validParams();

  CoordinatePlaneNodalContactBC(const InputParameters & parameters);
  virtual bool shouldApply() override;
  virtual void timestepSetup() override { _iteration = 0; }
  virtual void jacobianSetup() override { _iteration++; }

protected:
  virtual Real computeQpValue() override;

private:
  unsigned _iteration;

  /// which half-space is the obstacle? True if the negative half-space is the obstacle.
  const bool _negative;

  /// The component (coordinate plane index)
  unsigned int _component;
};
