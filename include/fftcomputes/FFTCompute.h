//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "MooseObject.h"
#include "SwiftTypes.h"
#include "FFTBufferBase.h"

#include "torch/torch.h"

class FFTProblem;

/**
 * FFTCompute object
 */
class FFTCompute : public MooseObject
{
public:
  static InputParameters validParams();

  FFTCompute(const InputParameters & parameters);

protected:
  /// perform the computation
  virtual Real computeBuffer() = 0;

  torch::Tensor & getBuffer(const std::string & param);
  torch::Tensor & getBufferByName(const FFTBufferName & buffer_name);

  /// output buffer
  torch::Tensor & _u;

  FFTProblem & _fft_problem;
};
