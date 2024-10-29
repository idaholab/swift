//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "SecantSolver.h"
#include "TensorProblem.h"
#include "DomainAction.h"

registerMooseObject("SwiftApp", SecantSolver);

InputParameters
SecantSolver::validParams()
{
  InputParameters params = SplitOperatorBase::validParams();
  params.addClassDescription("Implicit secant solver time integration.");
  params.addParam<unsigned int>("substeps", 1, "secant solver substeps per time step.");
  params.addParam<unsigned int>("max_iterations", 5, "Maximum number of secant solver iteration.");
  params.addParam<Real>("relative_tolerance", 1e-9, "Convergence tolerance.");
  params.addParam<Real>("absolute_tolerance", 1e-9, "Convergence tolerance.");
  params.addParam<Real>("damping", 1.0, "Damping factor for teh update step.");
  params.addParam<Real>(
      "dt_epsilon", 1e-4, "Semi-implicit stable timestep to bootstrap secant solve.");
  params.set<unsigned int>("substeps") = 0;
  params.addParam<bool>("verbose", false, "Show convergence history.");
  return params;
}

SecantSolver::SecantSolver(const InputParameters & parameters)
  : SplitOperatorBase(parameters),
    IterativeTensorSolverInterface(),
    _substeps(getParam<unsigned int>("substeps")),
    _max_iterations(getParam<unsigned int>("max_iterations")),
    _relative_tolerance(getParam<Real>("relative_tolerance")),
    _absolute_tolerance(getParam<Real>("absolute_tolerance")),
    _verbose(getParam<bool>("verbose")),
    _damping(getParam<Real>("damping"))
{
  const auto n = _variables.size();
  if (n > 1)
    paramWarning("buffer",
                 "The secant solver only work well for uncoupled variables. Use the BroydenSolver "
                 "for solves with multiple coupled variables.");
}

void
SecantSolver::computeBuffer()
{
  for (_substep = 0; _substep < _substeps; ++_substep)
    secantSolve();
}

void
SecantSolver::secantSolve()
{
  const auto n = _variables.size();
  const auto dt = _dt / _substeps;
  std::vector<torch::Tensor> u_old(n);
  std::vector<torch::Tensor> Rprev(n);
  std::vector<torch::Tensor> uprev(n);
  std::vector<Real> R0norm(n);

  // initial guess computed using semi-implicit Euler
  for (auto & cmp : _computes)
    cmp->computeBuffer();

  for (const auto i : make_range(n))
  {
    auto & u_out = _variables[i]._buffer;
    const auto & u = _variables[i]._reciprocal_buffer;
    const auto & N = _variables[i]._nonlinear_reciprocal;
    const auto & L = _variables[i]._linear_reciprocal;

    Rprev[i] = (N + L * u) * dt; // u = u_old at this point!
    uprev[i] = u;

    R0norm[i] = torch::norm(Rprev[i]).item<double>();

    // previous timestep solution
    if (_variables[i]._reciprocal_buffer.defined())
      u_old[i] = _variables[i]._reciprocal_buffer;
    else
      u_old[i] = _domain.fft(_variables[i]._buffer);

    // now modify u_out
    const auto dt_epsilon = getParam<Real>("dt_epsilon");
    u_out = _domain.ifft((u + dt_epsilon * N) / (1.0 - dt_epsilon * L));

    if (_verbose)
      std::cout << "|R0|=" << R0norm[i] << std::endl;
  }

  // Jacobian
  torch::Tensor J;
  // Residual
  torch::Tensor R;

  // secant iterations
  bool all_converged;
  for (_iterations = 0; _iterations < _max_iterations; ++_iterations)
  {
    // evaluate the solve computes
    for (auto & cmp : _computes)
      cmp->computeBuffer();

    all_converged = true;

    // integrate all variables
    for (const auto i : make_range(n))
    {
      auto & u_out = _variables[i]._buffer;
      const auto & u = _variables[i]._reciprocal_buffer;
      const auto & N = _variables[i]._nonlinear_reciprocal;
      const auto & L = _variables[i]._linear_reciprocal;

      // residual in reciprocal space
      R = (N + L * u) * dt + u_old[i] - u;

      // avoid NaN
      const auto dx = u - uprev[i];
      const auto dy = R - Rprev[i];
      auto du = torch::where(dy != 0, -R * dx / dy, 0.0);

      uprev[i] = u;
      Rprev[i] = R;

      if (_damping == 1.0)
        u_out = _domain.ifft(u + du);
      else
        u_out = _domain.ifft(u + du * _damping);

      const auto Rnorm = torch::norm(R).item<double>();

      if (_verbose)
      {
        const auto unorm = torch::norm(du).item<double>();
        std::cout << _iterations << " |du| = " << unorm << " |R|=" << Rnorm << std::endl;
      }

      // nan check
      if (std::isnan(Rnorm))
      {
        all_converged = false;
        _iterations = _max_iterations;
        std::cout << "NaN detected, aborting solve.\n";
        break;
      }

      // relative convergence check
      all_converged =
          all_converged && (Rnorm < _absolute_tolerance || Rnorm / R0norm[i] < _relative_tolerance);
    }

    if (all_converged)
    {
      // std::cout << "Secant solve converged after " << _iterations << " iterations. |R|=" <<Rnorm
      // << " |R|/|R0|=" << Rnorm / R0norm << '\n';
      _is_converged = true;
      break;
    }
  }

  if (!all_converged)
  {
    std::cout << "Solve not converged.\n";

    // restore old solution (TODO: fix time, etc)
    for (const auto i : make_range(n))
      _variables[i]._buffer = _domain.ifft(u_old[i]);

    _is_converged = false;
  }
}
