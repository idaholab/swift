
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "PerformFFT.h"
#include "DomainAction.h"

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
    _u = _domain.fft(_input);
  else
    _u = _domain.ifft(_input);
}

template class PerformFFTTempl<true>;
template class PerformFFTTempl<false>;
