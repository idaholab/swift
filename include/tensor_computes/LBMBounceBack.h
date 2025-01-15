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
 * LBMBounceBack object
 */
class LBMBounceBack : public LBMBoundaryCondition
{
public:
  static InputParameters validParams();

  LBMBounceBack(const InputParameters & parameters);

  void topBoundary() override;
  void bottomBoundary() override;
  void leftBoundary() override;
  void rightBoundary() override;
  void frontBoundary() override;
  void backBoundary() override;
  void wallBoundary() override;
  void computeBuffer() override;

protected:
  const std::vector<torch::Tensor> &  _f_old;
  const std::array<int64_t, 3> _grid_size;
};
