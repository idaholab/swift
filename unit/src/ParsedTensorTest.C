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

  auto a = MooseFFT::create2DBuffer(5, 5);
  auto b = MooseFFT::create2DBuffer(5, 5);
  auto c = MooseFFT::create2DBuffer(5, 5);

  auto & A = a.data();
  auto & B = b.data();
  auto & C = b.data();

  auto [x, y] = a.getAxis();

  // function
  A = x;
  B = y;
  C = -x * x;

  std::vector<neml2::Scalar> params;

  std::cout << F.customEval(params.data()) << '\n';
}

#else

#warning "NEML2 not found, skipping FFTBuffer unit test."

#endif
