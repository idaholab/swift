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
#include "FFTBuffer.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>

TEST(ParsedTensorTest, Parse)
{
  ParsedTensor F;
  std::string variables = "a, b, c";

  F.Parse("a * b + c", variables);
  F.Optimize();
  F.setupTensors();

  auto a = MooseFFT::createBuffer({5, 5}, {6.0, 6.0});
  auto b = MooseFFT::createBuffer({5, 5}, {6.0, 6.0});
  auto c = MooseFFT::createBuffer({5, 5}, {6.0, 6.0});

  auto & A = a.data();
  auto & B = b.data();
  auto & C = c.data();

  auto [x, y] = a.getAxis();

  // function
  A = x;
  B = y;
  C = -x * y;

  std::vector<neml2::Scalar> params{A, B, C};

  // Frobenius norm of the result tensor
  EXPECT_NEAR(
      torch::linalg::norm(F.Eval(params.data()), "fro", {}, false, {}).item<double>(), 0.0, 1e-12);
}

#else

#warning "NEML2 not found, skipping FFTBuffer unit test."

#endif
