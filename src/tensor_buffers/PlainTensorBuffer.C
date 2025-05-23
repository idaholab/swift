/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "PlainTensorBuffer.h"

registerMooseObject("SwiftApp", PlainTensorBuffer);

InputParameters
PlainTensorBuffer::validParams()
{
  InputParameters params = TensorBuffer<torch::Tensor>::validParams();
  return params;
}

PlainTensorBuffer::PlainTensorBuffer(const InputParameters & parameters)
  : TensorBuffer<torch::Tensor>(parameters)
{
}

void
PlainTensorBuffer::init()
{
  // _u = torch::zeros(_shape, _options);
}

void
PlainTensorBuffer::makeCPUCopy()
{
  if (_cpu_copy_requested)
  {
    try
    {
      if (_u.is_cpu())
        _u_cpu = _u.clone().contiguous();
      else
        _u_cpu = _u.cpu().contiguous();
    }
    catch (...)
    {
    }
  }
}
