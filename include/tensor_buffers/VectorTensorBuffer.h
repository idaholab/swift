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
 * Vector valued Tensor
 */
class VectorTensorBuffer : public TensorBufferBase
{
public:
  static InputParameters validParams();

  VectorTensorBuffer(const InputParameters & parameters);
};
