/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * Identity rank two tensor Tensor
 */
class RankTwoDiagonalTensor : public TensorOperator<>
{
public:
  static InputParameters validParams();

  RankTwoDiagonalTensor(const InputParameters & parameters);

  virtual void computeBuffer() override;

  /// mesh dimension
  const unsigned int & _dim;

  /// value on the diagonal
  const Real & _value;
};
