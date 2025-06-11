/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMConstantTensor.h"

registerMooseObject("SwiftApp", LBMConstantTensor);

InputParameters
LBMConstantTensor::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  params.addRequiredParam<std::vector<std::string>>("values", "The scalar constant names.");
  params.addClassDescription("LBMConstantTensor object.");
  return params;
}

LBMConstantTensor::LBMConstantTensor(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters)
{
}

void
LBMConstantTensor::init()
{
  auto names = getParam<std::vector<std::string>>("values");

  for (auto name : names)
  {
    const std::string coefficient_name = "C_" + name;
    const Real & coefficient = _lb_problem.getScalarConstant(coefficient_name);
    auto value = (_lb_problem.getScalarConstant(name)) / coefficient;
    _values.push_back(value);
  }
}

void
LBMConstantTensor::computeBuffer()
{
  if (_u.dim() > 3)
    for (int64_t i = 0; i < _values.size(); i++)
      _u.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), i})
          .fill_(_values[i]);
  else
    _u.fill_(_values[0]);
}
