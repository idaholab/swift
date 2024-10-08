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
#include "torch/torch.h"
#include <thread>

class TensorProblem;
class DomainAction;

/**
 * Direct buffer output
 */
class TensorOutput : public MooseObject
{
public:
  static InputParameters validParams();

  TensorOutput(const InputParameters & parameters);

  virtual void init() {}

  void startOutput();
  void waitForCompletion();

protected:
  virtual void output() = 0;

  TensorProblem & _tensor_problem;
  const DomainAction & _domain;

  std::thread _output_thread;

  /// The buffer this output object is outputting
  std::map<std::string, const torch::Tensor *> _out_buffers;
};
