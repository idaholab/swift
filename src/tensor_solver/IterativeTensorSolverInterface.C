//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "TensorSolver.h"
#include "TensorPredictor.h"
#include "TensorProblem.h"

IterativeTensorSolverInterface::IterativeTensorSolverInterface()
  :  _iterations(0),
    _is_converged(true)
{
}

void
IterativeTensorSolverInterface::applyPredictors()
{
  for (const auto & pred : _predictors)
    pred->computeBuffer();
}
