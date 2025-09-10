/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "RankTwoDiagonalTensor.h"
#include "SwiftUtils.h"
#include "DomainAction.h"

registerMooseObject("SwiftApp", RankTwoDiagonalTensor);

InputParameters
RankTwoDiagonalTensor::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Rank two identity tensor in real space.");
  params.addParam<Real>("value", 1.0, "Value on the diagonal");
  return params;
}

RankTwoDiagonalTensor::RankTwoDiagonalTensor(const InputParameters & parameters)
  : TensorOperator(parameters), _dim(_domain.getDim()), _value(getParam<Real>("value"))
{
}

void
RankTwoDiagonalTensor::computeBuffer()
{
  const auto full = _domain.getValueShape({_dim, _dim});
  _u = (torch::eye(_dim, MooseTensor::floatTensorOptions()) * _value).expand(full);
}
