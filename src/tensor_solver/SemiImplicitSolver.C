/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "SemiImplicitSolver.h"
#include "TensorProblem.h"
#include "Conversion.h"
#include "DomainAction.h"
#include <array>

registerMooseObject("SwiftApp", SemiImplicitSolver);

namespace
{
// Max order supported (up to ABM4)
constexpr std::size_t max_order = 4;
}

InputParameters
SemiImplicitSolver::validParams()
{
  InputParameters params = SplitOperatorBase::validParams();
  params.addClassDescription("Semi-implicit time integration solver.");
  params.addParam<unsigned int>("substeps", 1, "semi-implicit substeps per time step.");
  params.addRangeCheckedParam<std::size_t>("order",
                                           2,
                                           "order > 0 & order <= " + Moose::stringify(max_order),
                                           "semi-implicit substeps per time step.");
  return params;
}

SemiImplicitSolver::SemiImplicitSolver(const InputParameters & parameters)
  : SplitOperatorBase(parameters),
    _substeps(getParam<unsigned int>("substeps")),
    _order(getParam<std::size_t>("order") - 1),
    _sub_dt(_tensor_problem.subDt()),
    _sub_time(_tensor_problem.subTime())
{
}

void
SemiImplicitSolver::computeBuffer()
{
  // Adams–Bashforth coefficients (zero-padded)
  constexpr std::array<std::array<double, max_order>, max_order> beta = {{
      {1.0, 0.0, 0.0, 0.0},                                  // AB1
      {3.0 / 2.0, -1.0 / 2.0, 0.0, 0.0},                     // AB2
      {23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0, 0.0},          // AB3
      {55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0}, // AB4
  }};

  // Adams–Moulton coefficients (zero-padded)
  constexpr std::array<std::array<double, max_order + 1>, max_order> alpha = {{
      {0.5, 0.5, 0.0, 0.0, 0.0},                                                    // AM1
      {5.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0, 0.0, 0.0},                              // AM2
      {9.0 / 24.0, 19.0 / 24.0, -5.0 / 24.0, 1.0 / 24.0, 0.0},                      // AM3
      {251.0 / 720.0, 646.0 / 720.0, -264.0 / 720.0, 106.0 / 720.0, -19.0 / 720.0}, // AM4
  }};

  const bool dt_changed = (_dt != _dt_old);

  torch::Tensor ubar;
  _sub_dt = _dt / _substeps;

  // subcycles
  for (const auto substep : make_range(_substeps))
  {
    // re-evaluate the solve compute
    _compute->computeBuffer();

    // integrate all variables
    for (auto & [u,
                 reciprocal_buffer,
                 linear_reciprocal,
                 nonlinear_reciprocal,
                 old_reciprocal_buffer,
                 old_non_linear_reciprocal] : _variables)
    {
      const auto n_old = std::min(old_reciprocal_buffer.size(), old_non_linear_reciprocal.size());

      // Order is what the uder requested, or what the available history allows for.
      // If dt changes between steps, we start at first order again
      const auto order = std::min(substep < _order && dt_changed ? 0 : n_old, _order);

      // Adams-Bashforth
      ubar = reciprocal_buffer + (_sub_dt * beta[order][0]) * nonlinear_reciprocal;
      for (const auto i : make_range(order))
        ubar += (_sub_dt * beta[order][i + 1]) * old_non_linear_reciprocal[i];
      ubar /= (1.0 - _sub_dt * linear_reciprocal);

      u = _domain.ifft(ubar);
    }

    if (substep < _substeps - 1)
      _tensor_problem.advanceState();

    // increment substep time
    _sub_time += _sub_dt;
  }
}
