//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTOutput.h"
#include "MooseError.h"
#include "SwiftTypes.h"
#include "FFTProblem.h"

InputParameters
FFTOutput::validParams()
{
  InputParameters params = MooseObject::validParams();
  params.addRequiredParam<std::vector<FFTInputBufferName>>("buffer", "The buffers to output");
  params.addParam<std::string>(
      "file_base",
      "The desired solution output name without an extension. If not provided, MOOSE sets it "
      "with Outputs/file_base when available. Otherwise, MOOSE uses input file name and this "
      "object name for a master input or uses master file_base, the subapp name and this object "
      "name for a subapp input to set it.");
  params.registerBase("FFTOutput");
  params.addPrivateParam<FFTProblem *>("_fft_problem", nullptr);
  params.addClassDescription("FFTOutput object.");
  return params;
}

FFTOutput::FFTOutput(const InputParameters & parameters)
  : MooseObject(parameters), _fft_problem(*getCheckedPointerParam<FFTProblem *>("_fft_problem"))
{
  for (const auto & name : getParam<std::vector<FFTInputBufferName>>("buffer"))
    _out_buffers[name] = &_fft_problem.getCPUBuffer(name);
}

void
FFTOutput::startOutput()
{
  if (_output_thread.joinable())
    mooseError("Output thread is already running. Must call waitForCompletion() first. This is a "
               "code error.");
  _output_thread = std::move(std::thread(&FFTOutput::output, this));
}

void
FFTOutput::waitForCompletion()
{
  if (_output_thread.joinable())
    _output_thread.join();
}
