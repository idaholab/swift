/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LBMBoundaryCondition.h"

/**
 * LBMFixedVelocityBC object
 */
template <int dimension>
class LBMFixedVelocityBCTempl : public LBMBoundaryCondition
{
public:
  static InputParameters validParams();

  LBMFixedVelocityBCTempl(const InputParameters & parameters);

  void topBoundary() override;
  void bottomBoundary() override;
  void leftBoundary() override;
  void rightBoundary() override;
  void frontBoundary() override;
  void backBoundary() override;
  void computeBuffer() override;

protected:
  const torch::Tensor & _f;
  const std::array<int64_t, 3> _grid_size;
  const double _velocity;
};

typedef LBMFixedVelocityBCTempl<2> LBMFixedVelocityBC2D;
typedef LBMFixedVelocityBCTempl<3> LBMFixedVelocityBC3D;
