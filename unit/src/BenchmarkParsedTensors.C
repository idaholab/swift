//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifdef NEML2_ENABLED

#include "ParsedTensor.h"
#include "ParsedJITTensor.h"
#include "FFTBuffer.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>
#include <chrono>

TEST(BenchmarkParsedTensors, Time)
{
  ParsedTensor F1;
  ParsedJITTensor F2;
  std::string variables = "a, b, c";

  const std::size_t N = 2000;
  auto A = MooseFFT::createBuffer({N, N}, {6.0, 6.0});
  auto B = MooseFFT::createBuffer({N, N});
  auto C = MooseFFT::createBuffer({N, N});

  auto & a = A.data();
  auto & b = B.data();
  auto & c = C.data();

  auto [x, y] = A.getAxis();

  // function
  a = x;
  b = y;
  c = 2.0 * x * y;

  F1.Parse("a * b + c + log(a+0.1) + a*a*a*a - b*b*b ", variables);
  F1.setupTensors();

  F2.Parse("a * b + c + log(a+0.1) + a*a*a*a - b*b*b ", variables);
  F2.setupTensors();

  // auto gold = a * b + c;

  if (!torch::cuda::is_available())
    mooseError("CUDA is not available! :-(");
  mooseInfo("CUDA DEVICE COUNT: ", torch::cuda::device_count());

  a.to(torch::kFloat32);
  b.to(torch::kFloat32);
  c.to(torch::kFloat32);
  a.to(torch::kCUDA);
  b.to(torch::kCUDA);
  c.to(torch::kCUDA);

  a.cuda();
  if (!a.is_cuda())
    mooseInfo("Tensor a should be on the GPU but is not");

  // time non-JIT
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  {
    std::vector<neml2::Scalar> params{a, b, c};
    if (!F1.Eval(params).is_cuda())
      mooseInfo("non-JIT result is not CUDA");
    auto t1 = high_resolution_clock::now();

    for (int i = 0; i < 1000; i++)
      F1.Eval(params);

    auto t2 = high_resolution_clock::now();

    MooseFFT::printTensorInfo(F1.Eval(params));

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;
    Moose::out << "non-JIT : " << ms_double.count() << "ms\n";
  }

  {
    std::vector<const torch::Tensor *> params{&a, &b, &c};
    if (!F2.Eval(params).is_cuda())
      mooseInfo("JIT result is not CUDA");
    auto t1 = high_resolution_clock::now();

    for (int i = 0; i < 1000; i++)
      F2.Eval(params);

    auto t2 = high_resolution_clock::now();

    MooseFFT::printTensorInfo(F2.Eval(params));

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;
    Moose::out << "    JIT : " << ms_double.count() << "ms\n";
  }
}

#else

#warning "NEML2 not found, skipping FFTBuffer unit test."

#endif
