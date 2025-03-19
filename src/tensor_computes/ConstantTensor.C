/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ConstantTensor.h"
#include "SwiftUtils.h"
#include "DomainAction.h"

registerMooseObject("SwiftApp", ConstantTensor);
registerMooseObject("SwiftApp", ConstantReciprocalTensor);

template <bool reciprocal>
InputParameters
ConstantTensorTempl<reciprocal>::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  if constexpr (reciprocal)
  {
    params.addClassDescription("Constant tensor in reciprocal space.");
    params.addParam<Real>("imaginary", 0.0, "Imaginary part of the constant value.");
  }
  else
    params.addClassDescription("Constant tensor in real space.");
  params.addParam<Real>("real", 0.0, "Real part of the constant value.");
  params.addParam<bool>("reciprocal", false, "Construct a reciprocal buffer");
  params.addParam<bool>("full", false, "Construct a full tensor will all entries");
  return params;
}

template <bool reciprocal>
ConstantTensorTempl<reciprocal>::ConstantTensorTempl(const InputParameters & parameters)
  : TensorOperator(parameters), _dim(_domain.getDim())
{
}

template <bool reciprocal>
void
ConstantTensorTempl<reciprocal>::computeBuffer()
{
  const auto real = this->getParam<Real>("real");
  if constexpr (reciprocal)
  {
    const auto & n = _domain.getLocalReciprocalGridSize();
    c10::IntArrayRef shape(n.data(), _dim);
    const auto imaginary = this->getParam<Real>("imaginary");
    _u = torch::complex(torch::full(shape, real, MooseTensor::floatTensorOptions()),
                        torch::full(shape, imaginary, MooseTensor::floatTensorOptions()));
  }
  else
  {
    const auto & n = _domain.getLocalGridSize();
    c10::IntArrayRef shape(n.data(), _dim);
    _u = torch::full(shape, real, MooseTensor::floatTensorOptions());
  }
}

template class ConstantTensorTempl<true>;
template class ConstantTensorTempl<false>;
