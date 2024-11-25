/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperatorBase.h"

/**
 * TensorOperator object with a single output
 */
class TensorOperator : public TensorOperatorBase
{
public:
  static InputParameters validParams();

  TensorOperator(const InputParameters & parameters);

protected:
  /// output buffer
  torch::Tensor & _u;
};
