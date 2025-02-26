/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ParsedTensor.h"
#include "ParsedJITTensor.h"
#include "SwiftUtils.h"
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
    ParsedTensor fp;
    ParsedJITTensor fp_jit;
    std::string variables = "x, y, n";
    if (fp.Parse(expression, variables) >= 0)
      mooseError("Invalid function: ", expression, "   ", fp.ErrorMsg());
    if (fp_jit.Parse(expression, variables) >= 0)
      mooseError("Invalid JIT function: ", expression, "   ", fp.ErrorMsg());

    // if (fp.AutoDiff(d) != -1)
    //   FAIL() << "Failed to take derivative w.r.t. " << d << " of " << expression << '\n';

    std::vector<const torch::Tensor *> params{&x, &y, &n};

    std::cout << "--  " << expression << std::endl;

    fp.setupTensors();
    fp_jit.setupTensors();

    const auto result_no_opt = fp.Eval(params);

    const auto result_jit_no_opt = fp_jit.Eval(params);

    fp.Optimize();
    fp_jit.Optimize();
    fp.setupTensors();
    fp_jit.setupTensors();

    const auto result_opt = fp.Eval(params);
    const auto result_jit_opt = fp_jit.Eval(params);

    EXPECT_NEAR(
        (result_no_opt - result_jit_no_opt).abs().max().template item<double>(), 0.0, 1e-12);
    EXPECT_NEAR((result_no_opt - result_opt).abs().max().template item<double>(), 0.0, 1e-12);
    EXPECT_NEAR((result_opt - result_jit_opt).abs().max().template item<double>(), 0.0, 1e-12);
    EXPECT_NEAR((result_opt - gold).abs().max().template item<double>(), 0.0, 1e-12);
  };

  // check cHypot opcode
  check("sqrt(x^2+y^2)", torch::sqrt(x * x + y * y));

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

  // check cRsqrt opcode
  check("1/sqrt(x+y)", 1.0 / torch::sqrt(x + y));

  // check cSincos opcode
  check("sin(y)-cos(y)", torch::sin(y) - torch::cos(y));

  // misc
  check("(x * y) / n", (x * y) / n);
  check("y/x", y / x);
  check("-x", -x);
  check("log(x)", torch::log(x));
  check("y^x", torch::pow(y, x));
  check("abs(y-x)", torch::abs(y - x));
  check("min(x,y)", torch::minimum(x, y));
  check("max(x,y)", torch::maximum(x, y));
  check("2^x", torch::pow(2, x));
  check("cbrt(x)", torch::pow(x, 1.0 / 3.0));
}
