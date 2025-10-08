/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ParsedJITTensor.h"
#include "SwiftUtils.h"
#include "MooseError.h"
#include "gtest/gtest.h"

#include <ATen/ops/linspace.h>
#include <string>
#include <vector>

TEST(ParsedTensorTest, Parse)
{
  const auto x =
      torch::unsqueeze(torch::linspace(0.1, 2, 10, MooseTensor::floatTensorOptions()), 1);
  const auto y = torch::unsqueeze(torch::linspace(0, 3, 15, MooseTensor::floatTensorOptions()), 0);
  const auto n = torch::max(x * y) * torch::tensor(1.01, MooseTensor::floatTensorOptions());

  auto check = [x, y, n](const std::string & expression, auto gold)
  {
    ParsedJITTensor fp_jit;
    std::vector<std::string> variables{"x", "y", "n"};

    if (!fp_jit.parse(expression, variables))
      mooseError("Invalid JIT function: ", expression, "   ", fp_jit.errorMessage());

    std::vector<const torch::Tensor *> params{&x, &y, &n};

    std::cout << "--  " << expression << std::endl;

    // Test without compilation
    const auto result_no_opt = fp_jit.eval(params);

    // Test with compilation (optimization)
    ParsedJITTensor fp_jit_opt;
    fp_jit_opt.parse(expression, variables);
    fp_jit_opt.compile(); // compile does optimization for JIT

    const auto result_opt = fp_jit_opt.eval(params);

    const Real epsilon =
        MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1e-12 : 1e-6;
    EXPECT_NEAR((result_no_opt - result_opt).abs().max().template item<double>(), 0.0, epsilon);
    EXPECT_NEAR((result_opt - gold).abs().max().template item<double>(), 0.0, epsilon);
  };

  // check hypot
  check("hypot(x,y)", torch::hypot(x, y));
  check("sqrt(x^2+y^2)", torch::sqrt(x * x + y * y));
  check("sqrt(x*x+y*y)", torch::sqrt(x * x + y * y));

  // trig functions
  check("cos(y)", torch::cos(y));
  check("sin(y)", torch::sin(y));
  check("cosh(y)", torch::cosh(y));
  check("sinh(y)", torch::sinh(y));
  check("atan(x + y)", torch::atan(x + y));
  check("asin((x * y) / n)", torch::asin((x * y) / n));
  check("acos((x * y) / n)", torch::acos((x * y) / n));
  check("acosh(x+y+1)", torch::acosh(x + y + 1));
  check("asinh(x-y)", torch::asinh(x - y));
  check("atan2(x,y)", torch::atan2(x, y));

  // check reciprocal sqrt
  check("1/sqrt(x+y)", 1.0 / torch::sqrt(x + y));

  // check sin and cos together
  check("sin(y)-cos(y)", torch::sin(y) - torch::cos(y));

  // misc
  check("(x * y) / n", (x * y) / n);
  check("y/x", y / x);
  check("-x", -x);
  check("log(x)", torch::log(x));
  check("log10(x)", torch::log10(x));
  check("log2(x)", torch::log2(x));
  check("pow(y, x)", torch::pow(y, x));
  check("abs(y-x)", torch::abs(y - x));
  check("floor(x-y)", torch::floor(x - y));
  check("ceil(x-y)", torch::ceil(x - y));
  check("round(x-y)", torch::round(x - y));
  check("trunc(x-y)", torch::trunc(x - y));
  check("min(x,y)", torch::minimum(x, y));
  check("max(x,y)", torch::maximum(x, y));
  check("pow(2, x)", torch::pow(2, x));
  check("pow(x, 1.0/3.0)", torch::pow(x, 1.0 / 3.0));
  // check("cbrt(x)", torch::pow(x, 1.0 / 3.0));
  check("if(x<1 | y>=2, x, y)", torch::where(torch::logical_or(x < 1, y >= 2), x, y));
  check("if(x<=2 & y>1, x*x, 3*y)", torch::where(torch::logical_and(x <= 2, y > 1), x*x, y*3));
}
