/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "MooseError.h"
#include "PlainTensorBuffer.h"
#include "SwiftUtils.h"
#include "DomainAction.h"

registerMooseObject("SwiftApp", PlainTensorBuffer);

InputParameters
PlainTensorBuffer::validParams()
{
  InputParameters params = TensorBuffer<torch::Tensor>::validParams();
  params.addParam<std::vector<int64_t>>("value_dimensions",{}, "Optional value dimensions");
  return params;
}

PlainTensorBuffer::PlainTensorBuffer(const InputParameters & parameters)
  : TensorBuffer<torch::Tensor>(parameters)
{
}

void
PlainTensorBuffer::init()
{
  _u = torch::zeros(_domain.getValueShape(getParam<std::vector<int64_t>>("value_dimensions")), MooseTensor::floatTensorOptions());
}

void
PlainTensorBuffer::makeCPUCopy()
{
  if (!_u.defined())
    return;

  if (_cpu_copy_requested)
  {
    if (_u.is_cpu())
      _u_cpu = _u.clone().contiguous();
    else
      _u_cpu = _u.cpu().contiguous();
  }
}
