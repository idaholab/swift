//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

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
