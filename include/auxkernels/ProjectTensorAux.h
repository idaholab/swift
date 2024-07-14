//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "AuxKernel.h"
#include "TensorProblemInterface.h"
#include "torch/torch.h"

class TensorProblem;

/**
 * Map an FFTBuffer to an AuxVariable
 */
class ProjectTensorAux : public AuxKernel, public TensorProblemInterface
{
public:
  static InputParameters validParams();

  ProjectTensorAux(const InputParameters & parameters);

protected:
  virtual Real computeValue() override;

  const torch::Tensor & _cpu_buffer;

  const unsigned int & _dim;
  const std::array<unsigned int, 3> & _n;
  const std::array<Real, 3> & _grid_spacing;
};
