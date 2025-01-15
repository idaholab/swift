/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LatticeBoltzmannOperator.h"
#include "MooseEnum.h"

/**
 * LBMBoundaryCondition object
 */
class LBMBoundaryCondition : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMBoundaryCondition(const InputParameters & parameters);

  virtual void topBoundary() = 0;
  virtual void bottomBoundary() = 0;
  virtual void leftBoundary() = 0;
  virtual void rightBoundary() = 0;
  virtual void frontBoundary() = 0;
  virtual void backBoundary() = 0;
  virtual void wallBoundary() = 0;
  virtual void computeBuffer() override;

protected:

  enum class Boundary
  {
    top,
    bottom,
    left,
    right,
    front,
    back,
    wall
  } _boundary;
};
