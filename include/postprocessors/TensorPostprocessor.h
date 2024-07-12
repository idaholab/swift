//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GeneralPostprocessor.h"
#include "torch/torch.h"

class FFTProblem;

/**
 * Postprocessor that operates on a buffer
 */
class FFTPostprocessor : public GeneralPostprocessor
{
public:
  static InputParameters validParams();

  FFTPostprocessor(const InputParameters & parameters);

protected:
  FFTProblem & _fft_problem;

  /// The buffer this postprocessor is operating on
  const torch::Tensor & _u;
};
