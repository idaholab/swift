//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifdef NEML2_ENABLED

#include "FFTBuffer.h"
#include "gtest/gtest.h"

TEST(FFTBuffer, Scalar)
{
  auto a = MooseFFT::create2DBuffer(10, 10);
  const auto pi = libMesh::pi;
  a.min() = {-pi, -pi};
  a.max() = {pi, 2 * pi};

  auto x = a.getAxis(0);
  auto y = a.getAxis(1);
  auto A = a.data();

  A = sin(2.0 * x) * sin(3.0 * y);
  auto A2 = 2.0 * cos(2.0 * x) * sin(3.0 * y);

  auto Abar = torch::fft::rfft2(A);

  // std::cout << Abar << '\n';

  auto [i, j] = a.getFrequencies();

  const auto I =
      neml2::Scalar(at::tensor(c10::complex<double>(0.0, 2.0 * pi), at::dtype(at::kComplexDouble)));

  // std::cout << i * I << '\n';
  // std::cout << j * I << '\n';

  std::cout << i << '\n';
  std::cout << j << '\n';

  // std::cout << neml2::Scalar(Abar) * (i * I) << '\n';
  // std::cout << neml2::Scalar(Abar) * (-I) << '\n';
  // std::cout << neml2::Scalar(Abar) * (i * I) << '\n';

  std::cout << torch::fft::irfft2(neml2::Scalar(Abar)) - A << '\n';
  // std::cout << torch::fft::irfft2(neml2::Scalar(Abar) * (i * I)) << '\n';
  // std::cout << A2 << '\n';

  std::cout << '\n';
  std::cout << torch::fft::irfft2(neml2::Scalar(Abar) * (i * I)) - A2 << '\n';
  // std::cout << A2 << '\n';

  // std::cout << torch::fft::irfft2(Abar) << '\n';
  // std::cout << torch::fft::irfft2(Abar) - A << '\n';
}

// TEST(FFTBuffer, Popel)
// {
//   auto a = MooseFFT::create2DBuffer(5, 6);
//   const auto pi = libMesh::pi;
//   a.min() = {0, 0};
//   a.max() = {2 * pi, 4 * pi};

//   auto x = a.getAxis(0, MooseFFT::Interval::RIGHT_OPEN);
//   auto y = a.getAxis(1, MooseFFT::Interval::RIGHT_OPEN);
//   auto A = a.data();

//   A = sin(x) + cos(y);
//   auto dAdx = cos(x);
//   auto dAdy = -sin(y);

//   auto Abar = torch::fft::rfft2(A);

//   std::cout << A << '\n';
//   std::cout << Abar << '\n';

//   auto [i, j] = a.getFrequencies();

//   const auto I =
//       neml2::Scalar(at::tensor(c10::complex<double>(0.0, 2.0 * pi),
//       at::dtype(at::kComplexDouble)));

//   std::cout << i << '\n';
//   std::cout << j << '\n';

//   // std::cout << neml2::Scalar(Abar) * (i * I) << '\n';
//   // std::cout << neml2::Scalar(Abar) * (-I) << '\n';
//   // std::cout << neml2::Scalar(Abar) * (i * I) << '\n';

//   // std::cout << torch::fft::irfft2(neml2::Scalar(Abar)) - A << '\n';
//   std::cout << torch::fft::irfft2(neml2::Scalar(Abar) * (i * I)) << '\n';
//   std::cout << torch::fft::irfft2(neml2::Scalar(Abar) * (j * I)) << '\n';

//   std::cout << dAdx << '\n';
//   std::cout << dAdy << '\n';

//   // std::cout << '\n';
//   // std::cout << torch::fft::irfft2(neml2::Scalar(Abar) * (i * I)) << '\n';
//   // std::cout << A2 << '\n';
//   // std::cout << torch::fft::irfft2(neml2::Scalar(Abar) * (i * I)) - A2 << '\n';
//   // std::cout << A2 << '\n';

//   // std::cout << torch::fft::irfft2(Abar) << '\n';
//   // std::cout << torch::fft::irfft2(Abar) - A << '\n';
// }

TEST(FFTBuffer, 2DAxis)
{
  auto a = MooseFFT::create2DBuffer(3, 4);
  a.min() = {0.0, 0.0};
  a.max() = {3.0 * 4.0 * 5.0, 4.0 * 5.0 * 6.0};

  const auto xCompare = [&a](auto interval, torch::detail::TensorDataContainer gold)
  {
    auto xt = a.getAxis(0, interval);
    auto xg = torch::unsqueeze(torch::tensor(gold), 1);
    EXPECT_TRUE(torch::equal(xt, xg));
  };

  const auto yCompare = [&a](auto interval, torch::detail::TensorDataContainer gold)
  {
    auto yt = a.getAxis(1, interval);
    auto yg = torch::unsqueeze(torch::tensor(gold), 0);
    EXPECT_TRUE(torch::equal(yt, yg));
  };

  xCompare(MooseFFT::Interval::CLOSED, {0.0, 30.0, 60.0});
  xCompare(MooseFFT::Interval::OPEN, {15.0, 30.0, 45.0});
  xCompare(MooseFFT::Interval::LEFT_OPEN, {20.0, 40.0, 60.0});
  xCompare(MooseFFT::Interval::RIGHT_OPEN, {0.0, 20.0, 40.0});

  yCompare(MooseFFT::Interval::CLOSED, {0.0, 40.0, 80.0, 120.0});
  yCompare(MooseFFT::Interval::OPEN, {24.0, 48.0, 72.0, 96.0});
  yCompare(MooseFFT::Interval::LEFT_OPEN, {30.0, 60.0, 90.0, 120.0});
  yCompare(MooseFFT::Interval::RIGHT_OPEN, {0.0, 30.0, 60.0, 90.0});
}

TEST(FFTBuffer, 1DAxis)
{
  auto a = MooseFFT::create1DBuffer(4);
  a.min() = {0.0};
  a.max() = {4.0 * 5.0 * 6.0};

  const auto xCompare = [&a](auto interval, torch::detail::TensorDataContainer gold)
  {
    auto xt = a.getAxis(0, interval);
    auto xg = torch::unsqueeze(torch::tensor(gold), 0);
    EXPECT_TRUE(torch::equal(xt, xg));
  };

  xCompare(MooseFFT::Interval::CLOSED, {0.0, 40.0, 80.0, 120.0});
  xCompare(MooseFFT::Interval::OPEN, {24.0, 48.0, 72.0, 96.0});
  xCompare(MooseFFT::Interval::LEFT_OPEN, {30.0, 60.0, 90.0, 120.0});
  xCompare(MooseFFT::Interval::RIGHT_OPEN, {0.0, 30.0, 60.0, 90.0});
}

#else

#warning "NEML2 not found, skipping FFTBuffer unit test."

#endif

