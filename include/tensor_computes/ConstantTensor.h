//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "TensorOperator.h"

/**
 * Constant Tensor
 */
template <bool reciprocal>
class ConstantTensorTempl : public TensorOperator
{
public:
  static InputParameters validParams();

  ConstantTensorTempl(const InputParameters & parameters);

  virtual void computeBuffer() override;

  /// mesh dimension
  const unsigned int & _dim;

  using TensorOperator::_domain;
  using TensorOperator::_u;
};

typedef ConstantTensorTempl<false> ConstantTensor;
typedef ConstantTensorTempl<true> ConstantReciprocalTensor;
