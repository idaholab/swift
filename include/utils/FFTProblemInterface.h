//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "MooseObject.h"
#include "FFTProblem.h"

class FFTProblemInterface
{
public:
  FFTProblemInterface(MooseObject * moose_object)
    : _fft_problem(
          [this, moose_object]()
          {
            auto fft_problem = dynamic_cast<FFTProblem *>(
                moose_object->parameters().getCheckedPointerParam<SubProblem *>("_subproblem"));
            if (!fft_problem)
              moose_object->mooseError(
                  "'", moose_object->name(), "' can only be used with FFTProblem");
            return std::ref(*fft_problem);
          }())
  {
  }

protected:
  FFTProblem & _fft_problem;
};
