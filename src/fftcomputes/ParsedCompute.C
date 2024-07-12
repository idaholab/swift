//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "ParsedCompute.h"

#include "MooseUtils.h"

registerMooseObject("SwiftApp", ParsedCompute);

InputParameters
ParsedCompute::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription("ParsedCompute object.");
  params.addRequiredParam<std::string>("expression", "Parsed expression");
  params.addParam<std::vector<FFTInputBufferName>>(
      "inputs", {}, "Buffer names used in the expression");
  params.addParam<std::vector<FFTInputBufferName>>(
      "derivatives", {}, "List of inputs to take the derivative w.r.t. (or none)");
  params.addParam<bool>(
      "enable_jit", true, "Use operator fusion and just in time compilation (recommended on GPU)");
  params.addParam<bool>("enable_fpoptimizer", true, "Use algebraic optimizer");
  params.addParam<bool>("extra_symbols",
                        true,
                        "Provide i (imaginary unit), j,k,l (reciprocal space frequency), x,y,z "
                        "(real space coordinates), and pi,e.");
  return params;
}

ParsedCompute::ParsedCompute(const InputParameters & parameters)
  : TensorOperator(parameters),
    _use_jit(getParam<bool>("enable_jit")),
    _extra_symbols(getParam<bool>("extra_symbols"))
{
  const auto & expression = getParam<std::string>("expression");
  const auto & names = getParam<std::vector<FFTInputBufferName>>("inputs");

  // get all input buffers
  for (const auto & name : names)
    _params.push_back(&getInputBufferByName(name));

  // build variable string
  auto variables = MooseUtils::join(names, ",");

  auto setup = [&](auto & fp)
  {
    // add extra symbols
    if (_extra_symbols)
    {
      variables = MooseUtils::join(std::vector<std::string>{variables, "i,x,j,y,k,z,l"}, ",");
      _constant_tensors.push_back(torch::tensor(c10::complex<double>(0.0, 1.0)));
      _params.push_back(&_constant_tensors[0]);

      for (const auto dim : make_range(3u))
      {
        _params.push_back(&_fft_problem.getAxis(dim));
        _params.push_back(&_fft_problem.getReciprocalAxis(dim));
      }

      fp.AddConstant("pi", libMesh::pi);
      fp.AddConstant("e", std::exp(Real(1.0)));
    }

    // parse
    fp.Parse(expression, variables);

    if (fp.Parse(expression, variables) >= 0)
      paramError("expression", "Invalid function: ", fp.ErrorMsg());

    // take derivatives
    for (const auto & d : getParam<std::vector<FFTInputBufferName>>("derivatives"))
      if (std::find(names.begin(), names.end(), d) != names.end())
      {
        if (fp.AutoDiff(d) != -1)
          paramError("expression", "Failed to take derivative w.r.t. `", d, "`.");
      }
      else
        paramError("derivatives",
                   "Derivative w.r.t `",
                   d,
                   "` was requested, but it is not listed in `inputs`.");

    if (getParam<bool>("enable_fpoptimizer"))
      fp.Optimize();

    fp.setupTensors();
  };

  if (_use_jit)
    setup(_jit);
  else
    setup(_no_jit);
}

void
ParsedCompute::computeBuffer()
{
  if (_use_jit)
    _u = _jit.Eval(_params);
  else
    _u = _no_jit.Eval(_params);
}
