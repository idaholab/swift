//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "AddFFTBufferAction.h"
#include "FFTProblem.h"

registerMooseAction("SwiftApp", AddFFTBufferAction, "add_fft_buffer");

InputParameters
AddFFTBufferAction::validParams()
{
  InputParameters params = MooseObjectAction::validParams();
  params.addClassDescription("Add an FFTBuffer object to the simulation.");
  params.set<std::string>("type") = "ScalarFFTBuffer";
  return params;
}

AddFFTBufferAction::AddFFTBufferAction(const InputParameters & parameters)
  : MooseObjectAction(parameters)
{
  mooseInfoRepeated("AddFFTBufferAction");
}

void
AddFFTBufferAction::act()
{
  mooseInfoRepeated("1 Adding buffer ", _name);

  auto fft_problem = std::dynamic_pointer_cast<FFTProblem>(_problem);
  if (!fft_problem)
    mooseError("FFT Buffers are only supported if the problem class is set to `FFTProblem`");

  fft_problem->addFFTBuffer(_name, _moose_object_pars);
}
