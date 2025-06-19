/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMTensorBuffer.h"
#include "DomainAction.h"
#include "LatticeBoltzmannStencilBase.h"
#include "LatticeBoltzmannProblem.h"

registerMooseObject("SwiftApp", LBMTensorBuffer);

InputParameters
LBMTensorBuffer::validParams()
{
  InputParameters params = TensorBuffer<torch::Tensor>::validParams();
  params.addRequiredParam<std::string>("buffer_type",
                                       "The buffer type can be either distribution function (df), "
                                       "macroscopic scaler (ms) or macroscopic vectorial (mv)");
  params.addPrivateParam<TensorProblem *>("_tensor_problem", nullptr);
  params.addClassDescription("Tensor wrapper form LBM tensors");

  return params;
}

LBMTensorBuffer::LBMTensorBuffer(const InputParameters & parameters)
  : TensorBuffer<torch::Tensor>(parameters),
    _buffer_type(getParam<std::string>("buffer_type")),
    _lb_problem(dynamic_cast<LatticeBoltzmannProblem &>(
        *getCheckedPointerParam<TensorProblem *>("_tensor_problem"))),
    _stencil(_lb_problem.getStencil())
{
}

void
LBMTensorBuffer::init()
{
  int64_t dimension = 0;
  if (_buffer_type == "df")
    dimension = _stencil._q;
  else if (_buffer_type == "mv")
    dimension = _domain.getDim();
  else if (_buffer_type == "ms")
    dimension = 1;
  else
    mooseError("Buffer type ", _buffer_type, " is not recognized");

  std::vector<int64_t> shape(_domain.getShape().begin(), _domain.getShape().end());

  if (_domain.getDim() < 3)
    shape.push_back(1);
  if (dimension > 0)
    shape.push_back(static_cast<int64_t>(dimension));

  _u = torch::zeros(shape, MooseTensor::floatTensorOptions());
}

void
LBMTensorBuffer::makeCPUCopy()
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
