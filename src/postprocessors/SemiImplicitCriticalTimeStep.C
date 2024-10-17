//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "SemiImplicitCriticalTimeStep.h"

registerMooseObject("SwiftApp", SemiImplicitCriticalTimeStep);

InputParameters
SemiImplicitCriticalTimeStep::validParams()
{
  InputParameters params = TensorPostprocessor::validParams();
  params.addClassDescription(
      "Compute the critical timestep given the reciprocal space representation of the linear "
      "operator in a semi-implicit time integrator.");
  params.addParam<Real>("c", 1.0, "Courant number (CFL factor)");
  return params;
}

SemiImplicitCriticalTimeStep::SemiImplicitCriticalTimeStep(const InputParameters & parameters)
  : TensorPostprocessor(parameters)
{
}

void
SemiImplicitCriticalTimeStep::execute()
{
  const auto max_norm_k = std::sqrt(torch::max(_u * _u.conj()).cpu().item<double>());
  _critical_dt = max_norm_k > 0.0 ? 1.0 / max_norm_k : 1e30;
}

void
SemiImplicitCriticalTimeStep::finalize()
{
  gatherMin(_critical_dt);
}

PostprocessorValue
SemiImplicitCriticalTimeStep::getValue() const
{
  return _critical_dt;
}
