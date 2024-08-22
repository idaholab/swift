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
#include "TensorBufferBase.h"

#include "torch/torch.h"

class TensorProblem;

/**
 * TensorOperatorBase object
 */
class TensorOperatorBase : public MooseObject, public DependencyResolverInterface
{
public:
  static InputParameters validParams();

  TensorOperatorBase(const InputParameters & parameters);

  virtual const std::set<std::string> & getRequestedItems() override { return _requested_buffers; }
  virtual const std::set<std::string> & getSuppliedItems() override { return _supplied_buffers; }

  /// perform the computation
  virtual void computeBuffer() = 0;

protected:
  const torch::Tensor & getInputBuffer(const std::string & param);
  const torch::Tensor & getInputBufferByName(const FFTInputBufferName & buffer_name);

  torch::Tensor & getOutputBuffer(const std::string & param);
  torch::Tensor & getOutputBufferByName(const FFTOutputBufferName & buffer_name);

  std::set<std::string> _requested_buffers;
  std::set<std::string> _supplied_buffers;

  TensorProblem & _tensor_problem;

  /// axes
  const torch::Tensor &_x, &_y, &_z;

  /// reciprocal axes
  const torch::Tensor &_i, &_j, &_k;
};
