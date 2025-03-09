/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#ifdef NEML2_ENABLED

#include "NEML2TensorBuffer.h"

registerMooseObject("SwiftApp", VectorTensor);
registerMooseObject("SwiftApp", SR2Tensor);

template <typename T>
InputParameters
NEML2TensorBuffer<T>::validParams()
{
  InputParameters params = TensorBuffer<T>::validParams();
  return params;
}

template <typename T>
NEML2TensorBuffer<T>::NEML2TensorBuffer(const InputParameters & parameters)
  : TensorBuffer<T>(parameters)
{
}

template <typename T>
void
NEML2TensorBuffer<T>::init()
{
  // TODO
}

template class NEML2TensorBuffer<neml2::Vec>;
template class NEML2TensorBuffer<neml2::SR2>;

#endif
