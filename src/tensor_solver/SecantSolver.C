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
  params.addParam<Real>("tolerance", 1e-10, "Convergence tolerance.");
  params.addParam<Real>(
      "dt_epsilon", 1e-4, "Semi-implicit stable timestep to bootstrap secant solve.");
  params.set<unsigned int>("substeps") = 0;
  params.addParam<bool>("verbose", false, "Show convergence history.");
  return params;
}

SecantSolver::SecantSolver(const InputParameters & parameters)
  : SplitOperatorBase(parameters),
    _substeps(getParam<unsigned int>("substeps")),
    _max_iterations(getParam<unsigned int>("max_iterations")),
    _tolerance(getParam<Real>("tolerance")),
    _verbose(getParam<bool>("verbose"))
{
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

  for (const auto i : make_range(n))
  {
    auto & u_out = _variables[i]._buffer;
    const auto & u = _variables[i]._reciprocal_buffer;
    const auto & N = _variables[i]._nonlinear_reciprocal;
    const auto & L = _variables[i]._linear_reciprocal;

    // initial guess computed using semi-implicit Euler
    for (auto & cmp : _computes)
      cmp->computeBuffer();
    Rprev[i] = (N + L * u) * dt; // u = u_old at this point!
    uprev[i] = u;

    // previous timestep solution
    if (_variables[i]._reciprocal_buffer.defined())
      u_old[i] = _variables[i]._reciprocal_buffer;
    else
      u_old[i] = _domain.fft(_variables[i]._buffer);

    // now modify u_out
    const auto dt_epsilon = getParam<Real>("dt_epsilon");
    u_out = _domain.ifft((u + dt_epsilon * N) / (1.0 - dt_epsilon * L));

    R0norm[i] = torch::sum(torch::abs(Rprev[i])).item<double>();
    if (_verbose)
      std::cout << "|R0|=" << R0norm[i] << std::endl;
  }

  // Jacobian
  torch::Tensor J;
  // Residual
  torch::Tensor R;

  // secant iterations
  bool all_converged;
  for (const auto iteration : make_range(_max_iterations))
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
      const auto denom = u - uprev[i];
      J = (R - Rprev[i]) / denom;
      auto du = torch::where(torch::abs(denom) > 0, -R / J, 0.0);

      uprev[i] = u;
      Rprev[i] = R;

      u_out = _domain.ifft(u + du);

      const auto Rnorm = torch::sum(torch::abs(R)).item<double>();

      if (_verbose)
      {
        const auto unorm = torch::sum(torch::abs(du)).item<double>();
        std::cout << iteration << " |du| = " << unorm << " |R|=" << Rnorm << std::endl;
      }

      // relative convergence check
      all_converged = all_converged && (Rnorm / R0norm[i] < _tolerance);
    }

    if (all_converged)
    {
      std::cout << "Secant solve converged after " << iteration << " iterations.\n";
      break;
    }
  }

  if (!all_converged)
    std::cout << "Solve not converged.\n";
}
