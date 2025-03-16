
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ReciprocalMatDiffusion.h"
#include "SwiftUtils.h"
#include <torch/torch.h>

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
  params.addRequiredParam<TensorInputBufferName>("psi", "Variable to impose Neuamnn BC.");
  params.addParam<Real>("epsilon",1e-8,"Epsilon to avoid divide by zero errors.");
  return params;
}

ReciprocalMatDiffusion::ReciprocalMatDiffusion(const InputParameters & parameters)
  : TensorOperator(parameters),
    _chem_pot(getInputBuffer("chemical_potential")),
    _M(getInputBuffer("mobility")),
    _psi(getInputBuffer("psi")),
    _epsilon(getParam<Real>("epsilon")),
    _imag(torch::tensor(c10::complex<double>(0.0, 1.0), MooseTensor::complexFloatTensorOptions()))
{
}

void
ReciprocalMatDiffusion::computeBuffer()
{
  auto psi_M = _M * (_psi > 0.0);
  auto J_x = psi_M * _domain.ifft(_i * _domain.fft(_chem_pot) * _imag);
  auto J_y = psi_M * _domain.ifft(_j * _domain.fft(_chem_pot) * _imag);
  auto J_z = psi_M * _domain.ifft(_k * _domain.fft(_chem_pot) * _imag);

  _u = _imag * (_i * _domain.fft(J_x) + _j * _domain.fft(J_y) +
                                     _k * _domain.fft(J_z));

  // auto div_J = _domain.ifft(_imag * (_i * _domain.fft(J_x) + _j * _domain.fft(J_y) +
  //                                    _k * _domain.fft(J_z)));
  // // auto result = div_J / (_psi + _epsilon);
  // // Create a mask where psi is zero
  // torch::Tensor mask = _psi == 0.0;
  // // // Perform the division where psi is not zero
  // torch::Tensor result = torch::zeros_like(div_J);
  // result.masked_scatter_(~mask, div_J.masked_select(~mask) / _psi.masked_select(~mask));
  // _u = div_J_hat; //result;//_domain.fft(div_J);
}