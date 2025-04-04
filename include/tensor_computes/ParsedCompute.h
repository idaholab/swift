/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"
#include "ParsedTensor.h"
#include "ParsedJITTensor.h"

/**
 * ParsedCompute object
 */
class ParsedCompute : public TensorOperator<>
{
public:
  static InputParameters validParams();

  ParsedCompute(const InputParameters & parameters);

  void computeBuffer() override;

protected:
  const bool _use_jit;
  const bool _extra_symbols;
  std::vector<torch::Tensor> _constant_tensors;

  ParsedJITTensor _jit;
  ParsedTensor _no_jit;

  torch::Tensor _time_tensor;
  std::vector<const torch::Tensor *> _params;
  const bool _real_space;
};
