/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMStackTensors.h"

registerMooseObject("SwiftApp", LBMStackTensors);

InputParameters
LBMStackTensors::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();

  params.addRequiredParam<std::vector<TensorInputBufferName>>("inputs", "Names of input tensor buffers to stack.");
  params.addClassDescription("Stack given scalar tensor buffers and output vectorial tensor.");

  return params;
}

LBMStackTensors::LBMStackTensors(const InputParameters  & parameters)
  : LatticeBoltzmannOperator(parameters)
{
}

void
LBMStackTensors::computeBuffer()
{ 
  using torch::indexing::Slice;

  const auto & names = getParam<std::vector<TensorInputBufferName>>("inputs");

  // check for duplicates
  auto hasDuplicates = [](const std::vector<std::string> & values)
  {
    std::set<std::string> s(values.begin(), values.end());
    return values.size() != s.size();
  };

  if (hasDuplicates(names))
    paramError("inputs", "Duplicate buffer name.");

  // make sure output buffer has the same dimensions
  if (_u.dim()<4)
    mooseError("Output buffer must be vectorial tensor.");

  std::vector<torch::Tensor> tensor_vector;
  for (const auto & name : names)
  {
      auto tensor_buffer = getInputBufferByName(name);
      
      if (tensor_buffer.dim() < 3)
          tensor_buffer = tensor_buffer.unsqueeze(2);
      if (tensor_buffer.dim() > 3)
      {
          std::string error_msg = "Input buffer ";
          error_msg.append(name);
          error_msg += " must be scalar";
          mooseError(error_msg);
      }
      tensor_vector.push_back(tensor_buffer);
  }

  // Stack the tensors along a new dimension
  _u = torch::stack(tensor_vector, 3);
}
