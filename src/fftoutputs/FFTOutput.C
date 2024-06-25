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

FFTOutput::FFTOutput(const FFTProblem & fft_problem) : _fft_problem(fft_problem) {}

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
