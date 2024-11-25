/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

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
