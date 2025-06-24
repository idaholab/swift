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
 * LBMFixedZerothOrderBC object
 */
template <int dimension>
class LBMFixedZerothOrderBCTempl : public LBMBoundaryCondition
{
public:
  static InputParameters validParams();

  LBMFixedZerothOrderBCTempl(const InputParameters & parameters);

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
  const Real _value;
};

typedef LBMFixedZerothOrderBCTempl<2> LBMFixedZerothOrderBC2D;
typedef LBMFixedZerothOrderBCTempl<3> LBMFixedZerothOrderBC3D;
