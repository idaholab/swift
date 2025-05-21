/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#ifdef NEML2_ENABLED

#include "TensorBuffer.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/SR2.h"

/**
 * Tensor wrapper arbitrary tensor value dimensions
 */
template <typename T>
class NEML2TensorBuffer : public TensorBuffer<T>
{
public:
  static InputParameters validParams();

  NEML2TensorBuffer(const InputParameters & parameters);

  virtual void init();
};

using VectorTensor = NEML2TensorBuffer<neml2::Vec>;
registerTensorType(VectorTensor, neml2::Vec);

using SR2Tensor = NEML2TensorBuffer<neml2::SR2>;
registerTensorType(SR2Tensor, neml2::SR2);

#endif
