/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MooseObject.h"
#include "SwiftTypes.h"
#include "DependencyResolverInterface.h"
#include "TensorBufferBase.h"

#include "torch/torch.h"

class TensorProblem;
class DomainAction;

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

  /// Helper to recursively update dependencies for grouped operators
  virtual void updateDependencies() {}

  /// perform the computation
  virtual void computeBuffer() = 0;

  /// called  after all objects have been constructed
  virtual void init() {}

  /// called if the simulation cell dimensions change
  virtual void gridChanged() {}

protected:
  const torch::Tensor & getInputBuffer(const std::string & param);
  const torch::Tensor & getInputBufferByName(const TensorInputBufferName & buffer_name);

  torch::Tensor & getOutputBuffer(const std::string & param);
  torch::Tensor & getOutputBufferByName(const TensorOutputBufferName & buffer_name);

  std::set<std::string> _requested_buffers;
  std::set<std::string> _supplied_buffers;

  TensorProblem & _tensor_problem;
  const DomainAction & _domain;

  /// axes
  const torch::Tensor &_x, &_y, &_z;

  /// reciprocal axes
  const torch::Tensor &_i, &_j, &_k;
};
