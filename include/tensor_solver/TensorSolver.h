//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "TensorOperator.h"
#include "TensorProblem.h"

/**
 * TensorTimeIntegrator object (this is mostly a compute object)
 */
class TensorSolver : public TensorOperatorBase
{
public:
  static InputParameters validParams();

  TensorSolver(const InputParameters & parameters);

  virtual void updateDependencies() override final;

protected:
  const std::vector<torch::Tensor> & getBufferOld(const std::string & param,
                                                  unsigned int max_states);
  const std::vector<torch::Tensor> & getBufferOldByName(const TensorInputBufferName & buffer_name,
                                                        unsigned int max_states);

  void gatherDependencies();

  const Real & _dt;
  const Real & _dt_old;

  /// root compute for teh solver
  std::shared_ptr<TensorOperatorBase> _compute;
};
