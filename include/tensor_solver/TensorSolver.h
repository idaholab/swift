/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

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

  /// root compute for the solver
  std::shared_ptr<TensorOperatorBase> _compute;
};
