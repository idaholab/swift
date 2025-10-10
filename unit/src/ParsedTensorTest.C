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
  // Use linspace values that avoid zero to prevent singularities in derivatives
  const auto x =
      torch::unsqueeze(torch::linspace(0.1, 2.01, 11, MooseTensor::floatTensorOptions()), 1);
  const auto y =
      torch::unsqueeze(torch::linspace(0.11, 3.02, 15, MooseTensor::floatTensorOptions()), 0);
  const auto n = torch::max(x * y) * 1.01;

  // Epsilon for precision-dependent testing
  const Real epsilon = MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1e-12 : 1e-6;

  // Epsilon tensor for finite difference calculations (use sqrt(epsilon) for better numerical
  // stability)
  const Real eps_fd = MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1e-6 : 1e-3;
  const Real rel_tolerance =
      MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1e-5 : 1e-2;
  const Real abs_tolerance = std::sqrt(epsilon);

  // Perturbed tensors for finite difference derivatives
  const auto x2 = x + eps_fd;
  const auto y2 = y + eps_fd;
  const auto n2 = n + eps_fd;

  auto check = [&](const std::string & expression, auto gold, bool derivative_check = true)
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

    EXPECT_NEAR((result_no_opt - result_opt).abs().max().template item<double>(), 0.0, epsilon);
    EXPECT_NEAR((result_opt - gold).abs().max().template item<double>(), 0.0, epsilon);

    if (!derivative_check)
      return;

    // Test derivatives using finite differences (commented out temporarily for debugging)
    std::vector<const torch::Tensor *> params_dx{&x2, &y, &n};
    std::vector<const torch::Tensor *> params_dy{&x, &y2, &n};
    std::vector<const torch::Tensor *> params_dn{&x, &y, &n2};

    auto derivativeTest = [&](const std::string & dvar, std::vector<const torch::Tensor *> dparams)
    {
      ParsedJITTensor dfp;
      if (dfp.parse(expression, variables))
      {
        const auto fp2 = fp_jit_opt.eval(dparams);
        const auto dr1 = (fp2 - result_opt) / eps_fd;

        dfp.differentiate(dvar);
        dfp.compile();
        const auto dr2 = dfp.eval(params);

        // Use relative error: |symbolic - fd| / (|fd| + eps) < tolerance
        const auto abs_diff = (dr2 - dr1).abs();
        const auto denominator = dr1.abs() + eps_fd;
        const auto rel_error = (abs_diff / denominator).max().template item<double>();

        // Relative tolerance: sqrt(epsilon) * 100 for first-order finite differences
        EXPECT_TRUE(rel_error < rel_tolerance ||
                    abs_diff.max().template item<double>() < abs_tolerance)
            << "Derivative w.r.t. " << dvar << "  failed for expression: " << expression << '\n'
            << "(" << rel_error << " < " << rel_tolerance << " || "
            << abs_diff.max().template item<double>() << " < " << abs_tolerance << ") is false";
      }
    };

    derivativeTest("x", params_dx);
    derivativeTest("y", params_dy);
    derivativeTest("n", params_dn);
  };

  auto check2 = [&](const std::string & expression,
                    std::string dvar,
                    const std::string & derivative,
                    bool compile)
  {
    std::vector<std::string> variables{"x", "y", "n"};
    std::vector<const torch::Tensor *> params{&x, &y, &n};

    ParsedJITTensor dF1, dF2;
    if (!dF1.parse(expression, variables))
      mooseError("Invalid JIT function: ", expression, "   ", dF1.errorMessage());
    if (!dF2.parse(derivative, variables))
      mooseError("Invalid JIT function: ", expression, "   ", dF2.errorMessage());

    dF1.differentiate(dvar);
    if (compile)
    {
      dF1.compile();
      dF2.compile();
    }

    const auto r1 = dF1.eval(params);
    const auto r2 = dF2.eval(params);

    const auto abs_diff = (r1 - r2).abs();
    const auto denominator = r1.abs() + epsilon;
    const auto rel_error = (abs_diff / denominator).max().template item<double>();

    EXPECT_NEAR(rel_error, 0.0, rel_tolerance);
  };

  // check hypot
  check("hypot(x,y)", torch::hypot(x, y));
  check("sqrt(x^2+y^2+n)", torch::sqrt(x * x + y * y + n));
  check("sqrt(x*x+y*y+n)", torch::sqrt(x * x + y * y + n));
  check("sqrt(x^2+y^2)", torch::sqrt(x * x + y * y));
  check("sqrt(x*x+y*y)", torch::sqrt(x * x + y * y));

  // trig functions
  check("tan((x-y)/2)", torch::tan((x - y) / 2.0));
  check("tanh(x-y)", torch::tanh(x - y));
  check("cos(y)", torch::cos(y));
  check("sin(y)", torch::sin(y));
  check("cosh(y)", torch::cosh(y));
  check("sinh(y)", torch::sinh(y));
  check("atan(x + y)", torch::atan(x + y));
  check("asin((x * y / 2) / n)", torch::asin((x * y / 2.0) / n));
  check("acos((x * y / 2) / n)", torch::acos((x * y / 2.0) / n));
  check("acosh(x+y+1)", torch::acosh(x + y + 1));
  check("asinh(x-y)", torch::asinh(x - y));
  check("atan2(x,y)", torch::atan2(x, y));

  // check reciprocal sqrt
  check("1/sqrt(x+y)", 1.0 / torch::sqrt(x + y));

  // check sin and cos together
  check("sin(y)-cos(y)", torch::sin(y) - torch::cos(y));

  // misc
  check("(x * y) / n", (x * y) / n);

  check("y/x", y / x, false);
  check2("y/x", "x", "-y/x^2", true);
  check2("y/x", "y", "1/x", true);

  check("-x", -x);
  check("rsqrt(x*y)", 1.0 / torch::sqrt(x * y));
  check("exp(x*y)", torch::exp(x * y));
  check("exp2(x*y)", torch::pow(2.0, x * y));
  check("(x*y) % 1.5", torch::remainder(x * y, 1.5));
  check("log(x)", torch::log(x));
  check("log10(x)", torch::log10(x));
  check("log2(x)", torch::log2(x));
  check("pow(y, x)", torch::pow(y, x));
  check("abs(y-x)", torch::abs(y - x), false);
  check("floor(x-y)", torch::floor(x - y), false);
  check("ceil(x-y)", torch::ceil(x - y), false);
  check("round(x-y)", torch::round(x - y), false);

  check("trunc(x-y)", torch::trunc(x - y));
  check("min(x^3,y^2)", torch::minimum(x * x * x, y * y));

  check("max(x^2,sin(4*y))", torch::maximum(x * x, torch::sin(y * 4.0)), false);
  check2("max(x^2,sin(4*y))", "x", "if(x^2>=sin(4*y),2*x,0)", true);
  check2("max(x^2,sin(4*y))", "y", "if(x^2>=sin(4*y),0,4*cos(4*y))", true);

  check("pow(2, x)", torch::pow(2, x));
  check("pow(x, 1.0/3.0)", torch::pow(x, 1.0 / 3.0));
  // check("cbrt(x)", torch::pow(x, 1.0 / 3.0));
  check("if(x<1 | y>=2, x, y)", torch::where(torch::logical_or(x < 1, y >= 2), x, y));
  check("if(x<=1 & y>2, x*x, 3*y)", torch::where(torch::logical_and(x <= 1, y > 2), x * x, y * 3));

  // local variables
  check("r2:=x^2+y^2; sqrt(r2)", torch::sqrt(x * x + y * y));
  check2("x2:=x^2; sinx2:=sin(x2); 4*sinx2", "x", "8*x*cos(x*x)", true);
  check2("a:=sin(x^2); a + 2*a + 3*a", "x", "12*x*cos(x^2)", true);
  check2("a:=sin(x^2); a + 2*a + 3*a", "x", "12*x*cos(x^2)", false);
}
