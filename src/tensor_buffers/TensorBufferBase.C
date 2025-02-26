/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorBufferBase.h"
#include "DomainAction.h"

registerMooseObjectAliased("SwiftApp", TensorBufferBase, "TensorBuffer");

InputParameters
TensorBufferBase::validParams()
{
  InputParameters params = MooseObject::validParams();
  params.addClassDescription("Generic TensorBuffer object.");
  params.registerBase("TensorBuffer");
  params.registerSystemAttributeName("TensorBuffer");
  params.addParam<bool>("reciprocal", false, "Is this a reciprocal space tensor?");
  params.addRangeCheckedParam<std::vector<int64_t>>(
      "value_shape", {}, "value_shape>0", "Shape of the tensor value entries.");
  params.addParam<std::vector<AuxVariableName>>(
      "map_to_aux_variable", {}, "Sync the given AuxVariable to the buffer contents");
  params.addParam<std::vector<AuxVariableName>>(
      "map_from_aux_variable", {}, "Sync the given AuxVariable to the buffer contents");
  return params;
}

TensorBufferBase::TensorBufferBase(const InputParameters & parameters)
  : torch::Tensor(),
    MooseObject(parameters),
    DomainInterface(this),
    _reciprocal(getParam<bool>("reciprocal")),
    _domain_shape(getParam<bool>("reciprocal") ? _domain.getReciprocalShape() : _domain.getShape()),
    _value_shape_buffer(getParam<std::vector<int64_t>>("value_shape")),
    _value_shape(_value_shape_buffer),
    _shape_buffer(
        [this]()
        {
          std::vector<int64_t> buffer;
          for (const auto d : _domain_shape)
            buffer.push_back(d);
          for (const auto v : _value_shape)
            buffer.push_back(v);
          return buffer;
        }()),
    _shape(_shape_buffer),
    _options(_reciprocal ? MooseTensor::complexFloatTensorOptions()
                         : MooseTensor::floatTensorOptions())
{
  const auto & map_to_aux_variable = getParam<std::vector<AuxVariableName>>("map_to_aux_variable");
  if (map_to_aux_variable.size() > 1)
    paramError("mapping to multiple variables is not supported.");

  if (!map_to_aux_variable.empty() && !_value_shape_buffer.empty())
    paramError("mapping non-scalar tensors is not supported.");

  if (!getParam<std::vector<AuxVariableName>>("map_from_aux_variable").empty())
    paramError("functionality is not yet implemented.");
}

TensorBufferBase &
TensorBufferBase::operator=(const torch::Tensor & rhs)
{
  if (this != &rhs)
  {
    torch::Tensor::operator=(rhs);
    expand();
  }
  return *this;
}

void
TensorBufferBase::expand()
{
  try
  {
    this->expand(_shape);
  }
  catch (const std::exception & e)
  {
    mooseError("Assignment of incompatible data to tensor '", MooseBase::name(), "'");
  }
}
