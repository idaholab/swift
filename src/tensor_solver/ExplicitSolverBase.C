/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ExplicitSolverBase.h"
#include "TensorProblem.h"
#include "DomainAction.h"

InputParameters
ExplicitSolverBase::validParams()
{
  InputParameters params = TensorSolver::validParams();
  params.addClassDescription("Base class for explicit time integrators.");

  params.addRequiredParam<std::vector<TensorOutputBufferName>>(
      "buffer", "The buffer this solver is writing to");

  params.addRequiredParam<std::vector<TensorInputBufferName>>(
      "reciprocal_buffer", "Buffer with the reciprocal of the integrated buffer");
  params.addRequiredParam<std::vector<TensorInputBufferName>>(
      "time_derivative_reciprocal", "Buffer with the reciprocal of the time derivative function");
  params.addParam<unsigned int>(
      "history_size", 1, "How many old states to use (determines time integration order).");
  return params;
}

ExplicitSolverBase::ExplicitSolverBase(const InputParameters & parameters)
  : TensorSolver(parameters), _history_size(getParam<unsigned int>("history_size"))
{
  auto buffers = getParam<std::vector<TensorOutputBufferName>>("buffer");
  auto reciprocal_buffers = getParam<std::vector<TensorInputBufferName>>("reciprocal_buffer");
  auto time_derivative_reciprocals =
      getParam<std::vector<TensorInputBufferName>>("time_derivative_reciprocal");

  const auto n = buffers.size();
  if (reciprocal_buffers.size() != n || time_derivative_reciprocals.size() != n)
    paramError("buffer",
               "Must have the same number of entries as 'reciprocal_buffer' and "
               "'time_derivative_reciprocal'.");

  for (const auto i : make_range(n))
    _variables.push_back(Variable{getOutputBufferByName(buffers[i]),
                                  getInputBufferByName(reciprocal_buffers[i]),
                                  getInputBufferByName(time_derivative_reciprocals[i]),
                                  getBufferOldByName(reciprocal_buffers[i], _history_size),
                                  getBufferOldByName(time_derivative_reciprocals[i], _history_size)});
}
