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

InputParameters
ParsedCompute::validParams()
{
  InputParameters params = FFTCompute::validParams();
  params.addClassDescription("ParsedCompute object.");
  params.addRequiredParam<std::string>("expession", "Parsed expression");
  params.addParam<std::vector<FFTInputBufferName>>("input_buffers", "Buffer names");
  params.addParam<bool>(
      "enable_jit", true, "Use operator fusion and just in time compilation (recommended on GPU)");
  return params;
}

ParsedCompute::ParsedCompute(const InputParameters & parameters)
  : FFTCompute(parameters), _use_jit(getParam<bool>("enable_jit"))
{
  const auto & names = getParam<std::vector<FFTInputBufferName>>("input_buffers");

  // get all input buffers
  for (const auto & name : names)
    _params.push_back(&getInputBuffer(name));

  // build variable string
  const auto variables = MooseUtils::join(names, ",");

  // parse
  const auto & expression = getParam<std::string>("expession");
  if (_use_jit)
  {
    _jit.Parse(expression, variables);
    _jit.setupTensors();
  }
  else
  {
    _no_jit.Parse(expression, variables);
    _no_jit.setupTensors();
  }
}

void
ParsedCompute::computeBuffer()
{
  if (_use_jit)
  {
    // _u = ;
  }
}
