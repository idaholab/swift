//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTElasticChemicalPotential.h"

registerMooseObject("SwiftApp", FFTElasticChemicalPotential);

InputParameters
FFTElasticChemicalPotential::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription("FFT based elastici strain energy chemical potential solve.");
  params.addParam<std::vector<FFTInputBufferName>>("displacements", "Displacements");
  params.addParam<FFTInputBufferName>("cbar", "FFT of concentration buffer");
  params.addRequiredParam<Real>("mu", "Lame mu");
  params.addRequiredParam<Real>("lambda", "Lame lambda");
  params.addRequiredParam<Real>("e0", "volumetric eigenstrain");
  return params;
}

FFTElasticChemicalPotential::FFTElasticChemicalPotential(const InputParameters & parameters)
  : TensorOperator(parameters),
    _two_pi_i(torch::tensor(c10::complex<double>(0.0, 2.0 * pi),
                            MooseFFT::floatTensorOptions().dtype(torch::kComplexDouble))),
    _mu(getParam<Real>("mu")),
    _lambda(getParam<Real>("lambda")),
    _e0(getParam<Real>("e0")),
    _cbar(getInputBuffer("cbar"))

{
  for (const auto & name : getParam<std::vector<FFTOutputBufferName>>("displacements"))
    _displacements.push_back(&getInputBufferByName(name));

  if (_fft_problem.getDim() != _displacements.size())
    paramError("displacements", "Need one displacement variable per mesh dimension");
}

void
FFTElasticChemicalPotential::computeBuffer()
{
  // wave vector
  const auto kx = _two_pi_i * _i;
  const auto ky = _two_pi_i * _j;
  const auto kz = _two_pi_i * _k;

  // FFT displacements
  auto ux = _fft_problem.fft(*_displacements[0]);
  auto uy = _fft_problem.fft(*_displacements[1]);
  auto uz = _fft_problem.fft(*_displacements[2]);

  // mu mech bar
  _u = -_e0 * (_e0 * (9.0 * _lambda * _cbar + _mu * 6.0 * _cbar) -
               (2.0 * _mu + 3.0 * _lambda) * (kx * ux + ky * uy + kz * uz));
}
