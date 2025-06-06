/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMTensorBuffer.h"
#include "DomainAction.h"

registerMooseObject("SwiftApp", LBMTensorBuffer);

InputParameters
LBMTensorBuffer::validParams()
{
  InputParameters params = TensorBuffer<torch::Tensor>::validParams();
  params.addParam<Real>("dimension", 0, "The vector dimension of tensor");
  params.addClassDescription("Tensor wrapper form LBM tensors");
    
  return params;
}

LBMTensorBuffer::LBMTensorBuffer(const InputParameters & parameters)
  : TensorBuffer<torch::Tensor>(parameters),
  _dimension(getParam<Real>("dimension"))
{
}

void 
LBMTensorBuffer::init()
{
  std::vector<int64_t> shape(_domain.getShape().begin(), _domain.getShape().end());
  if (_domain.getDim() < 3)
    shape.push_back(1);
  if (_dimension > 0)
    shape.push_back(static_cast<int64_t>(_dimension));
  _u = torch::zeros(shape, MooseTensor::floatTensorOptions());
}
