/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#ifdef NEML2_ENABLED

#include "TensorOperator.h"
#include "neml2/tensors/Vec.h"

/**
 * Gradient of a tensor field
 */
class GradientTensor : public TensorOperator<neml2::Vec>
{
public:
  static InputParameters validParams();

  GradientTensor(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _input;
  const bool _input_is_reciprocal;

  const torch::Tensor _zero;
};

#endif
