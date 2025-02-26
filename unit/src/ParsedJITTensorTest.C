/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ParsedJITTensor.h"
#include "gtest/gtest.h"

#include <ATen/ops/linspace.h>
#include <string>
#include <vector>

TEST(ParsedJITTensorTest, Parse)
{
  const auto x = torch::unsqueeze(torch::linspace(0, 2, 10), 1);
  const auto y = torch::unsqueeze(torch::linspace(0, 3, 15), 0);

  auto eval = [x, y](const std::string & expression)
  {
    ParsedJITTensor fp;
    std::string variables = "x, y";
    if (fp.Parse(expression, variables) >= 0)
      throw std::runtime_error("Invalid function: " + expression + "   " + fp.ErrorMsg());

    // if (fp.AutoDiff(d) != -1)
    //   FAIL() << "Failed to take derivative w.r.t. " << d << " of " << expression << '\n';

    std::vector<const torch::Tensor *> params{&x, &y};

    fp.setupTensors();

    const auto result_no_opt = fp.Eval(params);

    fp.Optimize();
    fp.setupTensors();

    const auto result_opt = fp.Eval(params);

    return std::make_pair(result_no_opt, result_opt);
  };

  auto check = [](auto a, auto b, auto c)
  {
    EXPECT_NEAR((a - b).abs().max().template item<double>(), 0.0, 1e-12);
    EXPECT_NEAR((a - c).abs().max().template item<double>(), 0.0, 1e-12);
  };

  { // check cHypot opcode
    const auto [a, b] = eval("sqrt(x^2+y^2)");
    check(a, b, torch::sqrt(x * x + y * y));
  }
}
