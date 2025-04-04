/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "MooseError.h"
#ifdef NEML2_ENABLED

#include "GradientTensor.h"
#include "SwiftUtils.h"
#include "TensorProblem.h"

registerMooseObject("SwiftApp", GradientTensor);

InputParameters
GradientTensor::validParams()
{
  InputParameters params = TensorOperator<neml2::Vec>::validParams();
  params.addClassDescription("Gradient of the coupled tensor buffer.");
  params.addParam<TensorInputBufferName>("input", "Input buffer name");
  params.addParam<bool>(
      "input_is_reciprocal", false, "Input buffer is already in reciprocal space");
  return params;
}

GradientTensor::GradientTensor(const InputParameters & parameters)
  : TensorOperator<neml2::Vec>(parameters),
    _input(getInputBuffer("input")),
    _input_is_reciprocal(getParam<bool>("input_is_reciprocal")),
    _zero(torch::tensor(0.0, MooseTensor::floatTensorOptions()))
{
}

void
GradientTensor::computeBuffer()
{
  auto i_reciprocal_input = (_input_is_reciprocal ? _input : _domain.fft(_input)) * _imaginary;
  auto grad_x = neml2::Scalar(_domain.ifft(i_reciprocal_input * _i), _dim);
  auto grad_y = neml2::Scalar(_dim > 1 ? _domain.ifft(i_reciprocal_input * _j) : _zero, _dim);
  auto grad_z = neml2::Scalar(_dim > 2 ? _domain.ifft(i_reciprocal_input * _k) : _zero, _dim);
  _u = neml2::Vec::fill(grad_x, grad_y, grad_z);
}

#endif
