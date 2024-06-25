//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "torch/torch.h"

class FFTProblem;

/**
 * Direct buffer output
 */
class FFTOutput
{
public:
  FFTOutput(const FFTProblem & fft_problem);

protected:
  const FFTProblem & _fft_problem;

  /// The buffer this postprocessor is operating on
  const std::vector<torch::Tensor *> & _out_buffers;
};
