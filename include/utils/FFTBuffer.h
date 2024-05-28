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
#include "neml2/tensors/Scalar.h"

#ifdef NEML2_ENABLED

using neml2::TorchShapeRef;

namespace MooseFFT
{

template <typename T, int D>
class FFTBuffer;

template <typename T = neml2::Scalar>
FFTBuffer<T, 1>
create1DBuffer(int nx, const torch::TensorOptions & options = neml2::default_tensor_options())
{
  return FFTBuffer<T, 1>(TorchShapeRef({nx}), options);
}

template <typename T = neml2::Scalar>
FFTBuffer<T, 2>
create2DBuffer(int nx,
               int ny,
               const torch::TensorOptions & options = neml2::default_tensor_options())
{
  return FFTBuffer<T, 2>(TorchShapeRef({nx, ny}), options);
}

template <typename T = neml2::Scalar>
FFTBuffer<T, 3>
create3DBuffer(int nx,
               int ny,
               int nz,
               const torch::TensorOptions & options = neml2::default_tensor_options())
{
  return FFTBuffer<T, 3>(TorchShapeRef({nx, ny, nz}), options);
}

enum class Interval
{
  CLOSED,
  OPEN,
  LEFT_OPEN,
  RIGHT_OPEN
};

template <typename T, int D>
class FFTBuffer
{
  FFTBuffer(const TorchShapeRef & batch_shape, const torch::TensorOptions & options);

public:
  T & data() { return _data; }
  const T & data() const { return _data; }

  std::array<neml2::Real, D> & min() { return _min; }
  std::array<neml2::Real, D> & max() { return _max; }

  neml2::Scalar getAxis(int dim, Interval interval = Interval::LEFT_OPEN) const;
  neml2::Scalar getFrequency(int dim) const;
  neml2::Scalar grad(int dim, T * fft = nullptr) const;

  auto getAxis() const { return getAxisHelper(std::make_integer_sequence<int, D>{}); }
  auto getFrequency() const { return getFrequencyHelper(std::make_integer_sequence<int, D>{}); }
  auto grad() const { return gradHelper(std::make_integer_sequence<int, D>{}); }

  neml2::Scalar laplace() const;

  // stream in gnuplot readable format

  friend FFTBuffer<T, 1> create1DBuffer<T>(int nx, const torch::TensorOptions & options);
  friend FFTBuffer<T, 2> create2DBuffer<T>(int nx, int ny, const torch::TensorOptions & options);
  friend FFTBuffer<T, 3>
  create3DBuffer<T>(int nx, int ny, int nz, const torch::TensorOptions & options);

  const neml2::Scalar _two_pi_i;

protected:
  T fft() const;
  T ifft(const T & Abar) const;

  template <int... dims>
  auto getAxisHelper(std::integer_sequence<int, dims...>) const
  {
    return std::make_tuple(getAxis(dims)...);
  }

  template <int... dims>
  auto getFrequencyHelper(std::integer_sequence<int, dims...>) const
  {
    return std::make_tuple(getFrequency(dims)...);
  }

  template <int... dims>
  auto gradHelper(std::integer_sequence<int, dims...>) const
  {
    return std::make_tuple(grad(dims)...);
  }

private:
  T _data;
  const bool _rfft;
  std::array<neml2::Real, D> _min;
  std::array<neml2::Real, D> _max;
};

template <typename T, int D>
FFTBuffer<T, D>::FFTBuffer(const TorchShapeRef & batch_shape, const torch::TensorOptions & options)
  : _two_pi_i(at::tensor(c10::complex<double>(0.0, 2.0 * pi), at::dtype(at::kComplexDouble))),
    _data(T::zeros(batch_shape, options)),
    _rfft(true)
{
  if (batch_shape.size() != D)
    throw std::domain_error("Invalid dimension");
}

template <typename T, int D>
neml2::Scalar
FFTBuffer<T, D>::getAxis(int dim, Interval interval) const
{
  if (dim < 0 || dim >= D)
    throw std::domain_error("Invalid dimension");

  const auto n = _data.batch_sizes()[dim];

  switch (interval)
  {
    case Interval::OPEN:
      return torch::unsqueeze(
          torch::narrow(
              neml2::Scalar::linspace(neml2::Scalar(_min[dim], neml2::default_tensor_options()),
                                      neml2::Scalar(_max[dim], neml2::default_tensor_options()),
                                      n + 2),
              0,
              1,
              n),
          D - dim - 1);

    case Interval::LEFT_OPEN:
      return torch::unsqueeze(
          torch::narrow(
              neml2::Scalar::linspace(neml2::Scalar(_min[dim], neml2::default_tensor_options()),
                                      neml2::Scalar(_max[dim], neml2::default_tensor_options()),
                                      n + 1),
              0,
              1,
              n),
          D - dim - 1);

    case Interval::RIGHT_OPEN:
      return torch::unsqueeze(
          torch::narrow(
              neml2::Scalar::linspace(neml2::Scalar(_min[dim], neml2::default_tensor_options()),
                                      neml2::Scalar(_max[dim], neml2::default_tensor_options()),
                                      n + 1),
              0,
              0,
              n),
          D - dim - 1);

    default: // case CLOSED:
      return torch::unsqueeze(
          neml2::Scalar::linspace(neml2::Scalar(_min[dim], neml2::default_tensor_options()),
                                  neml2::Scalar(_max[dim], neml2::default_tensor_options()),
                                  n),
          D - dim - 1);
  }
}

template <typename T, int D>
neml2::Scalar
FFTBuffer<T, D>::getFrequency(int dim) const
{
  if (dim < 0 || dim >= D)
    throw std::domain_error("Invalid dimension");

  const auto n = _data.batch_sizes()[dim];
  const auto a = (_max[dim] - _min[dim]) / Real(n);
  const auto freq = (dim == D - 1 && _rfft)
                        ? torch::fft::rfftfreq(n, a, neml2::default_tensor_options())
                        : torch::fft::fftfreq(n, a, neml2::default_tensor_options());

  return torch::unsqueeze(freq, D - dim - 1);
}

template <typename T, int D>
neml2::Scalar
FFTBuffer<T, D>::grad(int dim, T * cached_fft) const
{
  if (dim < 0 || dim >= D)
    throw std::domain_error("Invalid dimension");

  if (cached_fft)
    return ifft(*cached_fft * getFrequency(dim) * _two_pi_i);
  else
    return ifft(fft() * getFrequency(dim) * _two_pi_i);
}

template <typename T, int D>
T
FFTBuffer<T, D>::fft() const
{
  if constexpr (D == 1)

    return torch::fft::rfft2(_data);

  if constexpr (D == 2)
    return torch::fft::rfft2(_data);

  if constexpr (D == 2)
    throw std::domain_error("3D not implemented yet");
}

template <typename T, int D>
T
FFTBuffer<T, D>::ifft(const T & Abar) const
{
  if constexpr (D == 1)

    return torch::fft::irfft(Abar);

  if constexpr (D == 2)
    return torch::fft::irfft2(Abar);

  if constexpr (D == 2)
    throw std::domain_error("3D not implemented yet");
}

template <typename T, int D>
neml2::Scalar
FFTBuffer<T, D>::laplace() const
{
  neml2::Scalar cached_fft = fft();
  auto ret = grad(0, &cached_fft);
  ret *= ret;

  for (int dim = 1; dim < D; ++dim)
  {
    auto g = grad(dim, &cached_fft);
    ret += g * g;
  }
  return ret;
}
}

#endif
