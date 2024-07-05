
//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTQuasistaticElasticity.h"

registerMooseObject("SwiftApp", FFTQuasistaticElasticity);

InputParameters
FFTQuasistaticElasticity::validParams()
{
  InputParameters params = FFTComputeBase::validParams();
  params.addClassDescription("FFT based monolithic homogeneous quasistatic elasticity solve.");
  params.addParam<std::vector<FFTOutputBufferName>>("displacements", "Input buffer name");
  params.addRequiredParam<Real>("mu", "Lame mu");
  params.addRequiredParam<Real>("lambda", "Lame lambda");
  return params;
}

FFTQuasistaticElasticity::FFTQuasistaticElasticity(const InputParameters & parameters)
  : FFTComputeBase(parameters),
    _two_pi_i(torch::tensor(c10::complex<double>(0.0, 2.0 * pi),
                            MooseFFT::floatTensorOptions().dtype(torch::kComplexDouble))),
    _mu(getParam<Real>("mu")),
    _lambda(getParam<Real>("lambda"))

{
  for (const auto & name : getParam<std::vector<FFTOutputBufferName>>("displacements"))
    _displacements.push_back(&getOutputBufferByName(name));

  if (_fft_problem.getDim() != _displacements.size())
    paramError("displacements", "Need one displacemnet variable per mesh dimension");
}

void
FFTQuasistaticElasticity::computeBuffer()
{
  const auto & ux = *_displacements[0];
  const auto & uy = *_displacements[1];
  const auto & uz = *_displacements[2];

  // FFT displacements
  auto uxbar = _fft_problem.fft(ux);
  auto uybar = _fft_problem.fft(uy);
  auto uzbar = _fft_problem.fft(uz);

  // strain tensor (in reciprocal space)
  const auto exx = uxbar * _two_pi_i * _i;
  const auto eyy = uybar * _two_pi_i * _j;
  const auto ezz = uzbar * _two_pi_i * _k;
  const auto exy = 0.5 * (uxbar * _two_pi_i * _j + uybar * _two_pi_i * _i);
  const auto exz = 0.5 * (uxbar * _two_pi_i * _k + uzbar * _two_pi_i * _i);
  const auto eyz = 0.5 * (uybar * _two_pi_i * _k + uzbar * _two_pi_i * _j);
}
