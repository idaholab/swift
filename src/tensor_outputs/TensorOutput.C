//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "TensorOutput.h"
#include "MooseError.h"
#include "SwiftTypes.h"
#include "TensorProblem.h"
#include "DomainAction.h"

InputParameters
TensorOutput::validParams()
{
  InputParameters params = MooseObject::validParams();
  params.addRequiredParam<std::vector<TensorInputBufferName>>("buffer", "The buffers to output");
  params.addParam<std::string>(
      "file_base",
      "The desired solution output name without an extension. If not provided, MOOSE sets it "
      "with Outputs/file_base when available. Otherwise, MOOSE uses input file name and this "
      "object name for a master input or uses master file_base, the subapp name and this object "
      "name for a subapp input to set it.");
  params.registerBase("TensorOutput");
  params.addPrivateParam<TensorProblem *>("_tensor_problem", nullptr);
  params.addPrivateParam<const DomainAction *>("_domain", nullptr);
  params.addClassDescription("TensorOutput object.");
  return params;
}

TensorOutput::TensorOutput(const InputParameters & parameters)
  : MooseObject(parameters),
    _tensor_problem(*getCheckedPointerParam<TensorProblem *>("_tensor_problem")),
    _domain(*getCheckedPointerParam<const DomainAction *>("_domain"))
{
  for (const auto & name : getParam<std::vector<TensorInputBufferName>>("buffer"))
    _out_buffers[name] = &_tensor_problem.getCPUBuffer(name);
}

void
TensorOutput::startOutput()
{
  if (_output_thread.joinable())
    mooseError("Output thread is already running. Must call waitForCompletion() first. This is a "
               "code error.");
  _output_thread = std::move(std::thread(&TensorOutput::output, this));
}

void
TensorOutput::waitForCompletion()
{
  if (_output_thread.joinable())
    _output_thread.join();
}
