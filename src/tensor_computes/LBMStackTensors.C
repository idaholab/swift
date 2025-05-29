/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMStackTensors.h"

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

  // get all input buffers
  int dim_index = 0;
  for (const auto & name : names)
  {
    const auto tensor_buffer = getInputBufferByName(name);
    if (tensor_buffer.dim()>3)
    {
      std::string error_msg="Input buffer ";
      error_msg.append(name);
      error_msg+=" must be scalar";
      mooseError(error_msg);
    }
    _u.index_put_({Slice(), Slice(), Slice(), dim_index}, tensor_buffer);
    dim_index ++;
  }
}
