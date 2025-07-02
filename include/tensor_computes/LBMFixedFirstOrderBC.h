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
 * LBMFixedFirstOrderBC object
 */
template <int dimension>
class LBMFixedFirstOrderBCTempl : public LBMBoundaryCondition
{
public:
  static InputParameters validParams();

  LBMFixedFirstOrderBCTempl(const InputParameters & parameters);

  void init() override {};

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
  const Real & _value;
  const bool _perturb;
};

typedef LBMFixedFirstOrderBCTempl<9> LBMFixedFirstOrderBC9Q;
typedef LBMFixedFirstOrderBCTempl<19> LBMFixedFirstOrderBC19Q;
typedef LBMFixedFirstOrderBCTempl<27> LBMFixedFirstOrderBC27Q;
