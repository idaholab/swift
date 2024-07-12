//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTIntegralPostprocessor.h"
#include "FFTProblem.h"
#include "FFTMesh.h"

registerMooseObject("SwiftApp", FFTIntegralPostprocessor);

InputParameters
FFTIntegralPostprocessor::validParams()
{
  InputParameters params = FFTPostprocessor::validParams();
  params.addClassDescription("Compute the integral over a buffer");
  return params;
}

FFTIntegralPostprocessor::FFTIntegralPostprocessor(const InputParameters & parameters)
  : FFTPostprocessor(parameters)
{
}

void
FFTIntegralPostprocessor::execute()
{
  _integral = _u.sum().cpu().item<double>();

  const auto mesh = dynamic_cast<const FFTMesh *>(&_fft_problem.mesh());
  if (!mesh)
    mooseError("An FFTMesh is required");

  for (const auto dim : make_range(mesh->getDim()))
    _integral *= mesh->getMaxInDimension(dim);
}

PostprocessorValue
FFTIntegralPostprocessor::getValue() const
{
  return _integral;
}
