//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "TensorPredictor.h"
#include "TensorBuffer.h"
#include "TensorProblem.h"
#include "DomainAction.h"

InputParameters
TensorPredictor::validParams()
{
  InputParameters params = MooseObject::validParams();
  params.registerBase("TensorPredictor");
  params.addPrivateParam<TensorProblem *>("_tensor_problem", nullptr);
  params.addPrivateParam<const DomainAction *>("_domain", nullptr);
  params.addClassDescription("TensorPredictor object.");
  params.addRequiredParam<TensorOutputBufferName>("buffer",
                                                  "The buffer this compute is forward predicting");
  params.addParam<unsigned int>(
      "history_size", 1, "How many old states to use (determines time integration order).");
  return params;
}

TensorPredictor::TensorPredictor(const InputParameters & parameters)
  : MooseObject(parameters),
    _tensor_problem(*getCheckedPointerParam<TensorProblem *>("_tensor_problem")),
    _domain(*getCheckedPointerParam<const DomainAction *>("_domain")),
    _u_name(getParam<TensorOutputBufferName>("buffer")),
    _u(_tensor_problem.getBuffer(_u_name)),
    _u_old(_tensor_problem.getBufferOld(_u_name, getParam<unsigned int>("history_size")))
{
}
