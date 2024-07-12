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

/**
 * FFTTimeIntegrator object (this is mostly a compute object)
 */
class FFTTimeIntegrator : public TensorOperator
{
public:
  static InputParameters validParams();

  FFTTimeIntegrator(const InputParameters & parameters);

protected:
  const std::vector<torch::Tensor> & getBufferOld(const std::string & param,
                                                  unsigned int max_states);
  const std::vector<torch::Tensor> & getBufferOldByName(const FFTInputBufferName & buffer_name,
                                                        unsigned int max_states);

  const Real & _dt;
};
