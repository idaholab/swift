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
#include "TensorBuffer.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>

TEST(ParsedTensorTest, Parse)
{
  ParsedTensor F;
  std::string variables = "a, b, c";

  auto A = MooseTensor::createBuffer({5, 5}, {6.0, 6.0});
  auto B = MooseTensor::createBuffer({5, 5}, {6.0, 6.0});
  auto C = MooseTensor::createBuffer({5, 5}, {6.0, 6.0});

  auto & a = A.data();
  auto & b = B.data();
  auto & c = C.data();

  auto [x, y] = A.getAxis();

  // function
  a = x;
  b = y;
  c = 2.0 * x * y;

  F.Parse("a * b + c", variables);
  F.Optimize();
  F.setupTensors();

  std::vector<const torch::Tensor *> params{&a, &b, &c};
  auto gold = a * b + c;

  // compare parsed result tensor to compiled expression
  EXPECT_NEAR((F.Eval(params) - gold).abs().max().item<double>(), 0.0, 1e-12);
}

#else

#warning "NEML2 not found, skipping TensorBuffer unit test."

#endif
