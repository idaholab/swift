
//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "PerformFFT.h"

registerMooseObject("SwiftApp", ForwardFFT);
registerMooseObject("SwiftApp", InverseFFT);

template <bool forward>
InputParameters
PerformFFTTempl<forward>::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription("PerformFFT object.");
  params.addParam<TensorInputBufferName>("input", "Input buffer name");
  return params;
}

template <bool forward>
PerformFFTTempl<forward>::PerformFFTTempl(const InputParameters & parameters)
  : TensorOperator(parameters), _input(getInputBuffer("input"))
{
}

template <bool forward>
void
PerformFFTTempl<forward>::computeBuffer()
{
  if constexpr (forward)
    _u = _tensor_problem.fft(_input);
  else
    _u = _tensor_problem.ifft(_input);
}

template class PerformFFTTempl<true>;
template class PerformFFTTempl<false>;
