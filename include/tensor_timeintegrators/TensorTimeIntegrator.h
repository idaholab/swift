/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * TensorTimeIntegrator object (this is mostly a compute object)
 */
class TensorTimeIntegrator : public TensorOperator
{
public:
  static InputParameters validParams();

  TensorTimeIntegrator(const InputParameters & parameters);

protected:
  const std::vector<torch::Tensor> & getBufferOld(const std::string & param,
                                                  unsigned int max_states);
  const std::vector<torch::Tensor> & getBufferOldByName(const TensorInputBufferName & buffer_name,
                                                        unsigned int max_states);

  const Real & _sub_dt;
};
