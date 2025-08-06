/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

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
  params.addParam<unsigned int>("max_iterations", 30, "Maximum number of secant solver iteration.");
  params.addParam<Real>("relative_tolerance", 1e-9, "Convergence tolerance.");
  params.addParam<Real>("absolute_tolerance", 1e-9, "Convergence tolerance.");
  params.addParam<Real>("damping", 1.0, "Damping factor for the update step.");
  params.addParam<Real>("trust_radius", 1e20, "Maximum update norm.");
  params.addParam<bool>("adaptive_damping", true, "Update damping factor based on convergence.");
  params.addParam<Real>("adaptive_damping_cutback_factor", 0.5, "Multiply damping by this factor when the residual grows.");
  params.addParam<Real>("adaptive_damping_growth_factor", 1.2, "Multiply damping by this factor when the residual shrinks.");
  params.addParam<Real>(
      "dt_epsilon", 1e-4, "Semi-implicit stable timestep to bootstrap secant solve.");
  params.set<unsigned int>("substeps") = 1;
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
    _damping(getParam<Real>("damping")),
    _trust_radius(getParam<Real>("trust_radius")),
    _adaptive_damping(getParam<bool>("adaptive_damping")),
    _adaptive_damping_cutback_factor(getParam<Real>("adaptive_damping_cutback_factor")),
    _adaptive_damping_growth_factor(getParam<Real>("adaptive_damping_growth_factor")),
    _dt_epsilon(getParam<Real>("dt_epsilon")),
    _total_iterations(0)
{
  // no history required
  getVariables(0);

  const auto n = _variables.size();
  if (n > 1)
    paramWarning("buffer",
                 "The secant solver only work well for uncoupled variables. Use the BroydenSolver "
                 "for solves with multiple coupled variables.");
}

SecantSolver::~SecantSolver()
{
  if (_verbose)
    mooseInfo(_total_iterations, " total iterations.");
}

void
SecantSolver::computeBuffer()
{
  for (_substep = 0; _substep < _substeps; ++_substep)
    secantSolve();
}

void SecantSolver::secantSolve()
{
  const auto n = _variables.size();
  const auto dt = _dt / _substeps;
  std::vector<torch::Tensor> u_old(n);
  std::vector<torch::Tensor> Rprev(n);
  std::vector<torch::Tensor> uprev(n);
  std::vector<Real> R0norm(n);
  std::vector<Real> damping_factors(n, _damping); // Per-variable damping

  if (_verbose)
    _console << "Substep " << _substep << '\n';

  // initial guess using semi-implicit Euler
  _compute->computeBuffer();
  forwardBuffers();

  for (const auto i : make_range(n))
  {
    auto & u_out = _variables[i]._buffer;
    const auto & u = _variables[i]._reciprocal_buffer;
    const auto & N = _variables[i]._nonlinear_reciprocal;
    const auto * L = _variables[i]._linear_reciprocal;

    if (L)
      Rprev[i] = (N + *L * u) * dt;
    else
      Rprev[i] = N * dt;

    uprev[i] = u;
    R0norm[i] = torch::norm(Rprev[i]).item<double>();

    u_old[i] = u.defined() ? u : _domain.fft(_variables[i]._buffer);

    if (L)
      u_out = _domain.ifft((u + _dt_epsilon * N) / (1.0 - _dt_epsilon * *L));
    else
      u_out = _domain.ifft(u + _dt_epsilon * N);

    if (_verbose)
      _console << "|R0|=" << R0norm[i] << std::endl;
  }

  // forward predict (on solver outputs)
  applyPredictors();

  torch::Tensor R;

  // secant iterations
  bool all_converged;
  for (_iterations = 0; _iterations < _max_iterations; ++_iterations)
  {
    // re-evaluate the solve compute
    _compute->computeBuffer();
    forwardBuffers();

    all_converged = true;

    // integrate all variables
    for (const auto i : make_range(n))
    {
      auto & u_out = _variables[i]._buffer;
      const auto & u = _variables[i]._reciprocal_buffer;
      const auto & N = _variables[i]._nonlinear_reciprocal;
      const auto * L = _variables[i]._linear_reciprocal;

      // residual in reciprocal space
      if (L)
        R = (N + *L * u) * dt + u_old[i] - u;
      else
        R = N * dt + u_old[i] - u;

      // avoid NaN
      const auto dx = u - uprev[i];
      const auto dy = R - Rprev[i];
      const Real epsilon = 1e-9;
      auto du = torch::where(torch::abs(dy) > epsilon, -R * dx / dy, 0.0);

      uprev[i] = u;
      auto Rnorm = torch::norm(R).item<double>();

      // Adaptive damping update
      if (_adaptive_damping && _iterations > 0)
      {
        const auto RprevNorm = torch::norm(Rprev[i]).item<double>();
        if (Rnorm > RprevNorm)
          damping_factors[i] *= _adaptive_damping_cutback_factor;
        else
          damping_factors[i] = std::min(1.0, damping_factors[i] * _adaptive_damping_growth_factor);

        damping_factors[i] = std::max(1e-3, damping_factors[i]);
      }

      Real unorm = 0.0;
      if (_verbose || _trust_radius < 1e20)
        unorm = torch::norm(du).item<double>();

      if (_trust_radius < 1e20 && unorm > _trust_radius)
      {
        du *= _trust_radius / unorm;
        unorm = _trust_radius;
      }

      Rprev[i] = R;
      if (damping_factors[i] != 1.0)
        u_out = _domain.ifft(u + damping_factors[i] * du);
      else
        u_out = _domain.ifft(u + du);

      if (_verbose)
      {
        _console << _iterations << " |du|=" << unorm << " |R|=" << Rnorm
                 << " damping=" << damping_factors[i] << std::endl;
      }

      if (!std::isfinite(Rnorm))
      {
        all_converged = false;
        _iterations = _max_iterations;
        _console << "NaN or Inf detected, aborting solve.\n";
        break;
      }

      all_converged = all_converged &&
                      (Rnorm < _absolute_tolerance || Rnorm / R0norm[i] < _relative_tolerance);
    }

    _total_iterations++;

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
    _console << "Solve not converged.\n";

    // restore old solution (TODO: fix time, etc)
    for (const auto i : make_range(n))
      _variables[i]._buffer = _domain.ifft(u_old[i]);

    _is_converged = false;
  }
}

