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
#include "TensorProblem.h"

class TensorProblemInterface
{
public:
  TensorProblemInterface(MooseObject * moose_object)
    : _tensor_problem(
          [moose_object]()
          {
            auto tensor_problem = dynamic_cast<TensorProblem *>(
                moose_object->parameters().getCheckedPointerParam<SubProblem *>("_subproblem"));
            if (!tensor_problem)
              moose_object->mooseError(
                  "'", moose_object->name(), "' can only be used with TensorProblem");
            return std::ref(*tensor_problem);
          }())
  {
  }

protected:
  TensorProblem & _tensor_problem;
};
