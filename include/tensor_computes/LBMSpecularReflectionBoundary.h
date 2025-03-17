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
 * LBM combination of bounce-back and specular reflection boundary condition
 */
class LBMSpecularReflectionBoundary : public LBMBoundaryCondition
{
public:
  static InputParameters validParams();

  LBMSpecularReflectionBoundary(const InputParameters & parameters);

  void buildBoundaryMask();
  void determineBoundaryTypes();
  void topBoundary() override {}
  void bottomBoundary() override {}
  void leftBoundary() override {}
  void rightBoundary() override {}
  void frontBoundary() override {}
  void backBoundary() override {}
  void wallBoundary() override;
  void computeBuffer() override;

protected:
  const std::vector<torch::Tensor> &  _f_old;
  torch::Tensor _boundary_types;
  torch::Tensor _specular_reflection_indices;
  torch::Tensor _boundary_mask;
  const Real _r; // combination coefficient
};
