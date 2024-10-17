//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "ReciprocalIntegral.h"

registerMooseObject("SwiftApp", ReciprocalIntegral);

InputParameters
ReciprocalIntegral::validParams()
{
  InputParameters params = TensorPostprocessor::validParams();
  params.addClassDescription("Extract the zero k-vector value (corresponding to the integral).");
  return params;
}

ReciprocalIntegral::ReciprocalIntegral(const InputParameters & parameters)
  : TensorPostprocessor(parameters)
{
}

void
ReciprocalIntegral::execute()
{
  static const at::indexing::TensorIndex zero[3] = {0, 0, 0};
  // Extract the zero-frequency component at index {0, 0, ..., 0} for arbitrary dimensions
  torch::Tensor zero_frequency_tensor =
      _u.index(torch::ArrayRef<at::indexing::TensorIndex>(zero, _u.dim()));

  // Convert to std::complex<double> (TODO: or float!)
  _integral = torch::real(zero_frequency_tensor).item<double>();

  // for parallelism only the proc owning the global 0 matters...
}

void
ReciprocalIntegral::finalize()
{
  // proc that owns 0 should broadcast it. everyone else listens
}

PostprocessorValue
ReciprocalIntegral::getValue() const
{
  return _integral;
}
