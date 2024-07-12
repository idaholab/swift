//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "AddFFTObjectAction.h"
#include "FFTProblem.h"

registerMooseAction("SwiftApp", AddFFTObjectAction, "add_fft_compute");
registerMooseAction("SwiftApp", AddFFTObjectAction, "add_fft_ic");
registerMooseAction("SwiftApp", AddFFTObjectAction, "add_fft_time_integrator");
registerMooseAction("SwiftApp", AddFFTObjectAction, "add_fft_output");

InputParameters
AddFFTObjectAction::validParams()
{
  InputParameters params = MooseObjectAction::validParams();
  params.addClassDescription("Add an TensorOperator object to the simulation.");
  return params;
}

AddFFTObjectAction::AddFFTObjectAction(const InputParameters & parameters)
  : MooseObjectAction(parameters)
{
}

void
AddFFTObjectAction::act()
{
  auto fft_problem = std::dynamic_pointer_cast<FFTProblem>(_problem);
  if (!fft_problem)
    mooseError("FFT objects are only supported if the problem class is set to `FFTProblem`");

  // use addObject<FFTxxxxxx>(_type, _name, _moose_object_pars, /* threaded = */ false) ?

  if (_current_task == "add_fft_compute")
    fft_problem->addFFTCompute(_type, _name, _moose_object_pars);

  if (_current_task == "add_fft_ic")
    fft_problem->addFFTIC(_type, _name, _moose_object_pars);

  if (_current_task == "add_fft_time_integrator")
    fft_problem->addFFTTimeIntegrator(_type, _name, _moose_object_pars);

  if (_current_task == "add_fft_output")
    fft_problem->addFFTOutput(_type, _name, _moose_object_pars);
}
