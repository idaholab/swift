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
registerMooseObject("SwiftApp", SSR4Tensor);

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
  mooseInfoRepeated("Instantiating NEML2 tensor class ", libMesh::demangle(typeid(T).name()));
}

template <typename T>
void
NEML2TensorBuffer<T>::init()
{
  // TODO
}

template class NEML2TensorBuffer<neml2::Vec>;
template class NEML2TensorBuffer<neml2::SR2>;
template class NEML2TensorBuffer<neml2::SSR4>;

#endif
