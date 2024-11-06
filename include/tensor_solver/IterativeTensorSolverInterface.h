//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

class TensorPredictor;

/**
 * Interface for tensor solvers with internal iterations
 */
class IterativeTensorSolverInterface
{
public:
  IterativeTensorSolverInterface();

  const unsigned int & getIterations() const  { return _iterations; }
  const bool & isConverged() const { return _is_converged; }

  void addPredictor(std::shared_ptr<TensorPredictor> predictor);

protected:
  void applyPredictors();

  unsigned int _iterations;
  bool _is_converged;

  std::vector<std::shared_ptr<TensorPredictor>> _predictors;
};
