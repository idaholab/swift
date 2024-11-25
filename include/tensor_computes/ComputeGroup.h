/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperatorBase.h"

/**
 * Compute group with internal dependency resolution
 */
class ComputeGroup : public TensorOperatorBase
{
public:
  static InputParameters validParams();

  ComputeGroup(const InputParameters & parameters);

  virtual void init() override;

  virtual void computeBuffer() override;

  virtual void updateDependencies() override;

protected:
  std::vector<std::shared_ptr<TensorOperatorBase>> _computes;
  bool _visited;
};
