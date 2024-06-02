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

template <typename T, std::size_t D>
class FFTBuffer;

template <typename T = neml2::Scalar, std::size_t D>
FFTBuffer<T, D>
createBuffer(long int const (&n)[D], Real const (&min)[D], Real const (&max)[D])
{
  TorchShapeRef nn(n);
  std::array<Real, D> amin, amax;
  for (std::size_t i = 0; i < D; ++i)
  {
    amin[i] = min[i];
    amax[i] = max[i];
  }

  return FFTBuffer<T, D>(nn, amin, amax);
}

template <typename T = neml2::Scalar, std::size_t D>
FFTBuffer<T, D>
createBuffer(long int const (&n)[D], Real const (&max)[D])
{
  Real min[D];
  for (std::size_t i = 0; i < D; ++i)
    min[i] = 0.0;
  return createBuffer(n, min, max);
}

template <typename T = neml2::Scalar, std::size_t D>
FFTBuffer<T, D>
createBuffer(long int const (&n)[D])
{
  Real max[D];
  for (std::size_t i = 0; i < D; ++i)
    max[i] = 1.0;
  return createBuffer(n, max);
}

enum class Interval
{
  CLOSED,
  OPEN,
  LEFT_OPEN,
  RIGHT_OPEN
};

template <typename T, std::size_t D>
class FFTBuffer
{
  FFTBuffer(const TorchShapeRef & batch_shape,
            const std::array<Real, D> & min,
            const std::array<Real, D> & max);

public:
  T & data() { return _data; }
  const T & data() const { return _data; }

  std::array<neml2::Real, D> & min() { return _min; }
  std::array<neml2::Real, D> & max() { return _max; }

  neml2::Scalar getAxis(std::size_t dim, Interval interval = Interval::RIGHT_OPEN) const;
  neml2::Scalar getFrequency(std::size_t dim) const;
  neml2::Scalar grad(std::size_t dim, T * fft = nullptr) const;

  auto getAxis() const { return getAxisHelper(std::make_integer_sequence<std::size_t, D>{}); }
  auto getFrequency() const
  {
    return getFrequencyHelper(std::make_integer_sequence<std::size_t, D>{});
  }
  auto grad() const { return gradHelper(std::make_integer_sequence<std::size_t, D>{}); }

  neml2::Scalar laplace() const;

  // stream in gnuplot readable format

  friend FFTBuffer<T, D>
  createBuffer<T, D>(long int const (&n)[D], Real const (&min)[D], Real const (&max)[D]);

  const neml2::Scalar _two_pi_i;

protected:
  T fft() const;
  T ifft(const T & Abar) const;

  template <std::size_t... dims>
  auto getAxisHelper(std::integer_sequence<std::size_t, dims...>) const
  {
    return std::make_tuple(getAxis(dims)...);
  }

  template <std::size_t... dims>
  auto getFrequencyHelper(std::integer_sequence<std::size_t, dims...>) const
  {
    return std::make_tuple(getFrequency(dims)...);
  }

  template <std::size_t... dims>
  auto gradHelper(std::integer_sequence<std::size_t, dims...>) const
  {
    return std::make_tuple(grad(dims)...);
  }

private:
  torch::Device _device;
  T _data;
  const bool _rfft;
  std::array<neml2::Real, D> _min;
  std::array<neml2::Real, D> _max;
};

template <typename T, std::size_t D>
FFTBuffer<T, D>::FFTBuffer(const TorchShapeRef & batch_shape,
                           const std::array<Real, D> & min,
                           const std::array<Real, D> & max)
  : _two_pi_i(at::tensor(c10::complex<double>(0.0, 2.0 * pi), at::dtype(at::kComplexDouble))),
    _device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
    _data(T::zeros(batch_shape, _device)),
    _rfft(true),
    _min(min),
    _max(max)
{
  if (batch_shape.size() != D)
    throw std::domain_error("Invalid dimension");
}

template <typename T, std::size_t D>
neml2::Scalar
FFTBuffer<T, D>::getAxis(std::size_t dim, Interval interval) const
{
  if (dim >= D)
    throw std::domain_error("Invalid dimension");

  const auto n = _data.batch_sizes()[dim];

  switch (interval)
  {
    case Interval::OPEN:
      return torch::unsqueeze(
          torch::narrow(torch::linspace(
                            c10::Scalar(_min[dim]), c10::Scalar(_max[dim]), n + 2, _device),
                        0,
                        1,
                        n),
          D - dim - 1);

    case Interval::LEFT_OPEN:
      return torch::unsqueeze(
          torch::narrow(
              torch::linspace(c10::Scalar(_min[dim]), c10::Scalar(_max[dim]), n + 1, _device),
              0,
              1,
              n),
          D - dim - 1);

    case Interval::RIGHT_OPEN:
      return torch::unsqueeze(
          torch::narrow(
              torch::linspace(c10::Scalar(_min[dim]), c10::Scalar(_max[dim]), n + 1, _device),
              0,
              0,
              n),
          D - dim - 1);

    default: // case CLOSED:
      return torch::unsqueeze(
          torch::linspace(c10::Scalar(_min[dim]), c10::Scalar(_max[dim]), n, _device), D - dim - 1);
  }
}

template <typename T, std::size_t D>
neml2::Scalar
FFTBuffer<T, D>::getFrequency(std::size_t dim) const
{
  if (dim >= D)
    throw std::domain_error("Invalid dimension");

  const auto n = _data.batch_sizes()[dim];
  const auto a = (_max[dim] - _min[dim]) / Real(n);
  const auto freq = (dim == D - 1 && _rfft) ? torch::fft::rfftfreq(n, a, _device)
                                            : torch::fft::fftfreq(n, a, _device);

  return torch::unsqueeze(freq, D - dim - 1);
}

template <typename T, std::size_t D>
neml2::Scalar
FFTBuffer<T, D>::grad(std::size_t dim, T * cached_fft) const
{
  if (dim >= D)
    throw std::domain_error("Invalid dimension");

  if (cached_fft)
    return ifft(*cached_fft * getFrequency(dim) * _two_pi_i);
  else
    return ifft(fft() * getFrequency(dim) * _two_pi_i);
}

template <typename T, std::size_t D>
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

template <typename T, std::size_t D>
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

template <typename T, std::size_t D>
neml2::Scalar
FFTBuffer<T, D>::laplace() const
{
  neml2::Scalar cached_fft = fft();
  auto ret = grad(0, &cached_fft);
  ret *= ret;

  for (std::size_t dim = 1; dim < D; ++dim)
  {
    auto g = grad(dim, &cached_fft);
    ret += g * g;
  }
  return ret;
}
}

#endif
