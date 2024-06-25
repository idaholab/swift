//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTProblem.h"

FFTRawXDMFOut::FFTRawXDMFOut(const FFTProblem & fft_problem)
  : FFTOutput(fft_problem), _output_thread()
{
}

void
start()
{
  _output_thread = std::move(std::thread(&FFTOutput::output, this));
}

void
wait()
{
  if (_output_thread.joinable())
    _output_thread.join();
}
