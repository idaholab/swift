/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "SemiImplicitSolver.h"
#include "TensorProblem.h"
#include "DomainAction.h"

registerMooseObject("SwiftApp", SemiImplicitSolver);

InputParameters
SemiImplicitSolver::validParams()
{
  InputParameters params = SplitOperatorBase::validParams();
  params.addClassDescription("Semi-implicit time integration solver.");
  params.addParam<unsigned int>("substeps", 1, "semi-implicit substeps per time step.");
  return params;
}

SemiImplicitSolver::SemiImplicitSolver(const InputParameters & parameters)
  : SplitOperatorBase(parameters), _substeps(getParam<unsigned int>("substeps"))
{
}

void
SemiImplicitSolver::computeBuffer()
{
  torch::Tensor ubar;
  const Real dt = _dt / _substeps;
  const bool dt_changed = (_dt != _dt_old);

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

      if (n_old == 0 || (substep == 0 && dt_changed))
        // compute FFT time update (1st order)
        ubar = (reciprocal_buffer + dt * nonlinear_reciprocal) / (1.0 - dt * linear_reciprocal);

      if (n_old >= 1)
        // compute FFT time update (2nd order) - this probably breaks for adaptive dt!
        // ubar = (4.0 * _reciprocal_buffer - _old_reciprocal_buffer[0] +
        //         (2.0 * dt) * (2.0 * _non_linear_reciprocal - _old_non_linear_reciprocal[0])) /
        //        (3.0 - (2.0 * dt) * _linear_reciprocal);
        ubar = (reciprocal_buffer +
                dt / 2.0 * (3.0 * nonlinear_reciprocal - old_non_linear_reciprocal[0])) /
               (1.0 - dt * linear_reciprocal);

      u = _domain.ifft(ubar);
    }

    if (substep < _substeps - 1)
      _tensor_problem.advanceState();
  }
}
