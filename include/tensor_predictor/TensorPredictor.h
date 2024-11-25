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

#include "torch/torch.h"

class TensorProblem;
class DomainAction;

/**
 * TensorPredictor object
 */
class TensorPredictor : public MooseObject
{
public:
  static InputParameters validParams();

  TensorPredictor(const InputParameters & parameters);

  /// perform the computation
  virtual void computeBuffer() = 0;

  /// called if the simulation cell dimensions change
  virtual void gridChanged() {}

protected:
  TensorProblem & _tensor_problem;
  const DomainAction & _domain;

  const TensorOutputBufferName & _u_name;

  /// output buffer
  torch::Tensor & _u;

  /// old states of the output buffer
  const std::vector<torch::Tensor> & _u_old;
};
