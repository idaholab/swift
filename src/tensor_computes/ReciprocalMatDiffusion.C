
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ReciprocalMatDiffusion.h"
#include "SwiftUtils.h"

registerMooseObject("SwiftApp", ReciprocalMatDiffusion);

InputParameters
ReciprocalMatDiffusion::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription(
      "Calculates the divergence of flux for a variable mobility in reciprocal space.");
  params.addRequiredParam<TensorInputBufferName>("chemical_potential",
                                                 "Chemical potential buffer name");
  params.addRequiredParam<TensorInputBufferName>("mobility", "Mobility buffer name");
  return params;
}

ReciprocalMatDiffusion::ReciprocalMatDiffusion(const InputParameters & parameters)
  : TensorOperator(parameters),
    _chem_pot(getInputBuffer("chemical_potential")),
    _M(getInputBuffer("mobility")),
    _imag(torch::tensor(c10::complex<double>(0.0, 1.0), MooseTensor::complexFloatTensorOptions()))
{
}

void
ReciprocalMatDiffusion::computeBuffer()
{
  auto J_x = _M * _domain.ifft(_i * _domain.fft(_chem_pot) * _imag);
  auto J_y = _M * _domain.ifft(_j * _domain.fft(_chem_pot) * _imag);
  auto J_z = _M * _domain.ifft(_k * _domain.fft(_chem_pot) * _imag);

  _u = _imag * (_i * _domain.fft(J_x) + _j * _domain.fft(J_y) + _k * _domain.fft(J_z));
}