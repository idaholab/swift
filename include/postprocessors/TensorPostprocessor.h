/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "GeneralPostprocessor.h"
#include "DomainInterface.h"
#include "torch/torch.h"

class TensorProblem;

/**
 * Postprocessor that operates on a buffer
 */
class TensorPostprocessor : public GeneralPostprocessor, public DomainInterface
{
public:
  static InputParameters validParams();

  TensorPostprocessor(const InputParameters & parameters);

protected:
  TensorProblem & _tensor_problem;

  /// The buffer this postprocessor is operating on
  const torch::Tensor & _u;
};
