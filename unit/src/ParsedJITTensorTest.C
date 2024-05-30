//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifdef NEML2_ENABLED

#include "ParsedJITTensor.h"
#include "FFTBuffer.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>

TEST(ParsedJITTensorTest, Parse)
{
  ParsedJITTensor F;
  std::string variables = "a, b, c";

  F.Parse("a * b + c", variables);
  // F.Optimize();
  F.setupTensors();
}

#else

#warning "NEML2 not found, skipping FFTBuffer unit test."

#endif
