/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorOperatorBase.h"
#include "TensorBuffer.h"
#include "TensorProblem.h"
#include "DomainAction.h"

InputParameters
TensorOperatorBase::validParams()
{
  InputParameters params = MooseObject::validParams();
  params.registerBase("TensorOperator");
  params.addPrivateParam<TensorProblem *>("_tensor_problem", nullptr);
  params.addPrivateParam<const DomainAction *>("_domain", nullptr);
  params.addClassDescription("TensorOperatorBase object.");
  return params;
}

TensorOperatorBase::TensorOperatorBase(const InputParameters & parameters)
  : MooseObject(parameters),
    _requested_buffers(),
    _supplied_buffers(),
    _tensor_problem(*getCheckedPointerParam<TensorProblem *>("_tensor_problem")),
    _domain(*getCheckedPointerParam<const DomainAction *>("_domain")),
    _x(_domain.getAxis(0)),
    _y(_domain.getAxis(1)),
    _z(_domain.getAxis(2)),
    _i(_domain.getReciprocalAxis(0)),
    _j(_domain.getReciprocalAxis(1)),
    _k(_domain.getReciprocalAxis(2)),
    _time(_tensor_problem.subTime())
{
}

const torch::Tensor &
TensorOperatorBase::getInputBuffer(const std::string & param)
{
  return getInputBufferByName(getParam<TensorInputBufferName>(param));
}

const torch::Tensor &
TensorOperatorBase::getInputBufferByName(const TensorInputBufferName & buffer_name)
{
  _requested_buffers.insert(buffer_name);
  return _tensor_problem.getBuffer(buffer_name);
}

torch::Tensor &
TensorOperatorBase::getOutputBuffer(const std::string & param)
{
  return getOutputBufferByName(getParam<TensorOutputBufferName>(param));
}

torch::Tensor &
TensorOperatorBase::getOutputBufferByName(const TensorOutputBufferName & buffer_name)
{
  _supplied_buffers.insert(buffer_name);
  return _tensor_problem.getBuffer(buffer_name);
}
