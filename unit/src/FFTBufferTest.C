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
  FFTBuffer<Scalar> a({16, 6});
  FFTBuffer<Scalar> b(16, 6);
}

#else
#warning "SHIT, no NEML2!"
#endif
