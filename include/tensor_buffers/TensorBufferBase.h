/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "SwiftUtils.h"
#include "MooseObject.h"
#include "InputParameters.h"
#include "DomainInterface.h"

/**
 * Tensor wrapper arbitrary tensor value dimensions
 */
class TensorBufferBase : public torch::Tensor, public MooseObject, public DomainInterface
{
public:
  static InputParameters validParams();

  TensorBufferBase(const InputParameters & parameters);

protected:
  const bool _reciprocal;

  const torch::IntArrayRef _domain_shape;

  const std::vector<int64_t> _value_shape_buffer;
  const torch::IntArrayRef _value_shape;

  const std::vector<int64_t> _shape_buffer;
  torch::IntArrayRef _shape;

  const torch::TensorOptions _options;
};
