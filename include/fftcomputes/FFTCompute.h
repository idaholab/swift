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
#include "DependencyResolverInterface.h"
#include "FFTBufferBase.h"

#include "torch/torch.h"

class FFTProblem;

/**
 * FFTCompute object
 */
class FFTCompute : public MooseObject, public DependencyResolverInterface
{
public:
  static InputParameters validParams();

  FFTCompute(const InputParameters & parameters);

  virtual const std::set<std::string> & getRequestedItems() override { return _requested_buffers; }
  virtual const std::set<std::string> & getSuppliedItems() override { return _supplied_buffers; }

protected:
  /// perform the computation
  virtual Real computeBuffer() = 0;

  const torch::Tensor & getInputBuffer(const std::string & param);
  const torch::Tensor & getInputBufferByName(const FFTInputBufferName & buffer_name);

  torch::Tensor & getOutputBuffer(const std::string & param);
  torch::Tensor & getOutputBufferByName(const FFTOutputBufferName & buffer_name);

  /// output buffer
  torch::Tensor & _u;

  FFTProblem & _fft_problem;

  std::set<std::string> _requested_buffers;
  std::set<std::string> _supplied_buffers;
};
