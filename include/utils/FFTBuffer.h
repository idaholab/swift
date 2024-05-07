//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "NEML2Utils.h"

#ifdef NEML2_ENABLED

template <typename T>
class FFTBuffer
{
public:
  FFTBuffer(const TorchShapeRef & batch_shape,
            const torch::TensorOptions & options = default_tensor_options());

  FFTBuffer(int... size, const torch::TensorOptions & options = default_tensor_options())
    : FFTBuffer(TorchShapeRef{size...}, options)
  {
  }

  T & data() { return _data; }
  const T & data() const { return _data; }

private:
  T _data;
};

template <typename T>
FFTBuffer::FFTBuffer(const TorchShapeRef & batch_shape,
                     const torch::TensorOptions & options = default_tensor_options())
  : _data(T::zeroes(batch_shape, options))
{
}

typedef FFTBuffer<Real> FFTRealBuffer;

#endif
