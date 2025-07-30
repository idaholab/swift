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
  params.addRequiredParam<std::vector<std::string>>("constants", "The scalar constant names.");
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
  auto names = getParam<std::vector<std::string>>("constants");

  for (auto name : names)
  {
    auto value = (_lb_problem.getConstant<Real>(name));
    _values.push_back(value);
  }
}

void
LBMConstantTensor::computeBuffer()
{
  if (_u.dim() > 3)
    for (const auto i : index_range(_values))
      _u.index({torch::indexing::Slice(),
                torch::indexing::Slice(),
                torch::indexing::Slice(),
                static_cast<int64_t>(i)})
          .fill_(_values[i]);
  else
    _u.fill_(_values[0]);
}
