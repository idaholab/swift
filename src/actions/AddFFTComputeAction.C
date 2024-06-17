//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "AddFFTComputeAction.h"
#include "FFTProblem.h"

registerMooseAction("SwiftApp", AddFFTComputeAction, "add_fft_compute");
registerMooseAction("SwiftApp", AddFFTComputeAction, "add_fft_ic");

InputParameters
AddFFTComputeAction::validParams()
{
  InputParameters params = MooseObjectAction::validParams();
  params.addClassDescription("Add an FFTCompute object to the simulation.");
  return params;
}

AddFFTComputeAction::AddFFTComputeAction(const InputParameters & parameters)
  : MooseObjectAction(parameters)
{
}

void
AddFFTComputeAction::act()
{
  auto fft_problem = std::dynamic_pointer_cast<FFTProblem>(_problem);

  if (_current_task == "add_fft_compute")
  {
    if (!fft_problem)
      mooseError("FFT Computes are only supported if the problem class is set to `FFTProblem`");

    // use addObject<FFTCompute>(_type, _name, _moose_object_pars, /* threaded = */ false)
    fft_problem->addFFTCompute(_type, _name, _moose_object_pars);
    return;
  }

  if (_current_task == "add_fft_ic")
  {
    if (!fft_problem)
      mooseError(
          "FFT initial conditions are only supported if the problem class is set to `FFTProblem`");

    // use addObject<FFTInitialCondition>(_type, _name, _moose_object_pars, /* threaded = */ false)
    fft_problem->addFFTIC(_type, _name, _moose_object_pars);
    return;
  }
}
