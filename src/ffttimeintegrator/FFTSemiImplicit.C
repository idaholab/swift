//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTSemiImplicit.h"
#include "FFTProblem.h"

InputParameters
FFTSemiImplicit::validParams()
{
  InputParameters params = FFTTimeIntegrator::validParams();
  params.addClassDescription("Semi-implicit time integrator.");
  params.addRequiredParam<FFTInputBufferName>(
      "reciprocal_buffer", "Buffer with the reciprocal of the integrated buffer");
  params.addRequiredParam<FFTInputBufferName>(
      "linear_reciprocal", "Buffer with the reciprocal of the linear prefactor (e.g. kappa*k^2)");
  params.addRequiredParam<FFTInputBufferName>(
      "nonlinear_reciprocal", "Buffer with the reciprocal of the non-linear contribution");
  return params;
}

FFTSemiImplicit::FFTSemiImplicit(const InputParameters & parameters)
  : FFTTimeIntegrator(parameters),
    _reciprocal_buffer(getInputBuffer("reciprocal_buffer")),
    _linear_reciprocal(getInputBuffer("linear_reciprocal")),
    _non_linear_reciprocal(getInputBuffer("nonlinear_reciprocal")),
    _old_reciprocal_buffer(getBufferOld("reciprocal_buffer", 2)),
    _old_non_linear_reciprocal(getBufferOld("nonlinear_reciprocal", 2))
{
}

void
FFTSemiImplicit::computeBuffer()
{
  // compute FFT time update (1st order)
  auto ubar =
      (_reciprocal_buffer + _dt * _non_linear_reciprocal) / (1.0 + _dt * _linear_reciprocal);

  _u = _fft_problem.ifft(ubar);
}
