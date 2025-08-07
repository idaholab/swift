/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "ApplyStabilization.h"

registerMooseObject("SwiftApp", ApplyStabilization);

InputParameters
ApplyStabilization::validParams()
{
  InputParameters params = TensorOperatorBase::validParams();
  params.addClassDescription("Apply stabilization to linear and nonlinear terms");

  params.addRequiredParam<TensorInputBufferName>("linear", "Unstabilized linear term.");
  params.addRequiredParam<TensorInputBufferName>("nonlinear", "Unstabilized nonlinear term.");

  params.addRequiredParam<TensorInputBufferName>(
      "reciprocal", "Reciprocal space variable time integrates in the equation.");
  params.addRequiredParam<TensorInputBufferName>("stabilization", "Stabilization term.");

  params.addParam<TensorOutputBufferName>("stabilized_linear", "Stabilized linear term.");
  params.addParam<TensorOutputBufferName>("stabilized_nonlinear", "Stabilized nonlinear term.");

  return params;
}

ApplyStabilization::ApplyStabilization(const InputParameters & params)
  : TensorOperatorBase(params),
    _linear(getInputBuffer("linear")),
    _nonlinear(getInputBuffer("nonlinear")),
    _reciprocal(getInputBuffer("reciprocal")),
    _stabilization(getInputBuffer("stabilization")),
    _stabilized_linear(
        isParamValid("stabilized_linear")
            ? getOutputBuffer("stabilized_linear")
            : getOutputBufferByName(getParam<TensorInputBufferName>("linear") + "_stabilized")),
    _stabilized_nonlinear(
        isParamValid("stabilized_nonlinear")
            ? getOutputBuffer("stabilized_nonlinear")
            : getOutputBufferByName(getParam<TensorInputBufferName>("nonlinear") + "_stabilized"))
{
}

void
ApplyStabilization::computeBuffer()
{
  _stabilized_linear = _linear + _stabilization;
  _stabilized_nonlinear = _nonlinear - _reciprocal * _stabilization;
}
