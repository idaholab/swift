/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorBufferBase.h"

class ScalarTensorBuffer : public TensorBufferBase
{
public:
  static InputParameters validParams();

  ScalarTensorBuffer(const InputParameters & parameters);
};
