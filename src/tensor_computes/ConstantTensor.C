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
  params.addParam<SwiftConstantName>("imaginary", "0.0", "Imaginary part of the constant value.");
  if constexpr (reciprocal)
    params.addClassDescription("Constant tensor in reciprocal space.");
  else
  {
    params.addClassDescription("Constant tensor in real space.");
    params.suppressParameter<SwiftConstantName>("imaginary");
  }
  params.addParam<SwiftConstantName>("real", "0.0", "Real part of the constant value.");
  params.addParam<bool>("reciprocal", false, "Construct a reciprocal buffer");
  params.addParam<bool>("full", false, "Construct a full tensor will all entries");
  return params;
}

template <bool reciprocal>
ConstantTensorTempl<reciprocal>::ConstantTensorTempl(const InputParameters & parameters)
  : TensorOperator(parameters),
    _dim(_domain.getDim()),
    _real(this->getConstant<Real>("real")),
    _imaginary(this->getConstant<Real>("imaginary"))
{
}

template <bool reciprocal>
void
ConstantTensorTempl<reciprocal>::computeBuffer()
{
  if constexpr (reciprocal)
    _u = torch::complex(
        torch::full(_domain.getReciprocalShape(), _real, MooseTensor::floatTensorOptions()),
        torch::full(_domain.getReciprocalShape(), _imaginary, MooseTensor::floatTensorOptions()));
  else
    _u = torch::full(_domain.getShape(), _real, MooseTensor::floatTensorOptions());
}

template class ConstantTensorTempl<true>;
template class ConstantTensorTempl<false>;
