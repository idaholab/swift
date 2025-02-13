/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorBufferBase.h"

/**
 * Symmetric rank two valued Tensor
 */
class SR2TensorBuffer : public TensorBufferBase
{
public:
  static InputParameters validParams();

  SR2TensorBuffer(const InputParameters & parameters);

  // NEML2::Scalar getNEML2() { return NEML2::Scalar(*this, _domain_shape, _value_shape); }
};
