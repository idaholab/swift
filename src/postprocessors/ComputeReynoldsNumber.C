/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ComputeReynoldsNumber.h"
#include "TensorProblem.h"
#include "TensorBuffer.h"

registerMooseObject("SwiftApp", ComputeReynoldsNumber);

InputParameters
ComputeReynoldsNumber::validParams()
{
  InputParameters params = TensorPostprocessor::validParams();
  params.addRequiredParam<std::string>("kinematic_viscosity", "Fluid viscosity");
  params.addRequiredParam<std::string>("diameter", "Characteristic diamaeter");
  params.addClassDescription("Compute Reynolds number.");
  return params;
}

ComputeReynoldsNumber::ComputeReynoldsNumber(const InputParameters & parameters)
  : TensorPostprocessor(parameters),
    _kinematic_viscosity(
        _tensor_problem.getScalarConstant(getParam<std::string>("kinematic_viscosity"))),
    _D(_tensor_problem.getScalarConstant(getParam<std::string>("diameter")))
{
}

void
ComputeReynoldsNumber::execute()
{
  const auto avg_speed =
      _u.sum().cpu().item<double>() / torch::numel(_u) * _tensor_problem.getScalarConstant("C_U");
  _Reynolds_number = avg_speed * _D / _kinematic_viscosity;
}

PostprocessorValue
ComputeReynoldsNumber::getValue() const
{
  return _Reynolds_number;
}
