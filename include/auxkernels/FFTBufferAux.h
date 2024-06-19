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
#include "torch/torch.h"

class FFTProblem;

/**
 * Map an FFTBuffer to an AuxVariable
 */
class FFTBufferAux : public AuxKernel
{
public:
  static InputParameters validParams();

  FFTBufferAux(const InputParameters & parameters);

  virtual void customSetup(const ExecFlagType & exec_type);

protected:
  virtual Real computeValue() override;

  FFTProblem * _fft_problem;

  const torch::Tensor & _buffer;
  torch::Tensor _cpu_buffer;

  const unsigned int & _dim;
  const std::array<unsigned int, 3> & _n;
  const std::array<Real, 3> & _grid_spacing;
};
