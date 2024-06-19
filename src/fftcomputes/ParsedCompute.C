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
  InputParameters params = FFTCompute::validParams();
  params.addClassDescription("ParsedCompute object.");
  params.addRequiredParam<std::string>("expression", "Parsed expression");
  params.addParam<std::vector<FFTInputBufferName>>(
      "inputs", {}, "Buffer names used in the expression");
  params.addParam<std::vector<FFTInputBufferName>>(
      "derivatives", {}, "List of inputs to take the derivative w.r.t. (or none)");
  params.addParam<bool>(
      "enable_jit", true, "Use operator fusion and just in time compilation (recommended on GPU)");
  params.addParam<bool>("enable_fpoptimizer", true, "Use algebraic optimizer");
  return params;
}

ParsedCompute::ParsedCompute(const InputParameters & parameters)
  : FFTCompute(parameters), _use_jit(getParam<bool>("enable_jit"))
{
  const auto & expression = getParam<std::string>("expression");
  const auto & names = getParam<std::vector<FFTInputBufferName>>("inputs");

  // get all input buffers
  for (const auto & name : names)
    _params.push_back(&getInputBufferByName(name));

  // build variable string
  const auto variables = MooseUtils::join(names, ",");

  auto setup = [&](auto & fp)
  {
    // parse
    fp.Parse(expression, variables);

    if (fp.Parse(expression, variables) >= 0)
      paramError("expression", "Invalid function: ", fp.ErrorMsg());

    // take derivatives
    for (const auto & d : getParam<std::vector<FFTInputBufferName>>("derivatives"))
      if (std::find(names.begin(), names.end(), d) != names.end())
      {
        if (fp.AutoDiff(d) != 1)
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
