//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTGradientSquare.h"
#include "SwiftUtils.h"

registerMooseObject("SwiftApp", FFTGradientSquare);

InputParameters
FFTGradientSquare::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription("Tensor gradient.");
  params.addRequiredParam<TensorInputBufferName>("input", "Input buffer name");
  params.addParam<bool>(
      "input_is_reciprocal", false, "Input buffer is already in reciprocal space");
  params.addParam<Real>("factor", 1.0, "Prefactor to the gradient square");
  params.addRequiredParam<MooseEnum>(
      "direction", MooseEnum("X=0 Y=1 Z=2"), "Which axis to take the gradient along.");
  return params;
}

FFTGradientSquare::FFTGradientSquare(const InputParameters & parameters)
  : TensorOperator(parameters),
    _input(getInputBuffer("input")),
    _input_is_reciprocal(getParam<bool>("input_is_reciprocal")),
    _factor(getParam<Real>("factor")),
    _dim(_domain.getDim()),
    _i(torch::tensor(c10::complex<double>(0.0, 1.0),
                     MooseTensor::floatTensorOptions().dtype(torch::kComplexDouble)))
{
}

void
FFTGradientSquare::computeBuffer()
{
  auto sqr = [](const auto & a) { return a * a; };

  const auto r = _input_is_reciprocal ? _input : _domain.fft(_input);

  _u = sqr(_domain.ifft(r * _domain.getReciprocalAxis(0) * _i));
  for (const auto d : make_range(1u, _dim))
    _u += sqr(_domain.ifft(r * _domain.getReciprocalAxis(d) * _i));

  if (_factor != 1.0)
    _u *= _factor;
}
