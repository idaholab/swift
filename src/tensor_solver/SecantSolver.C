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
  params.addRequiredParam<TensorOutputBufferName>("du_realspace", "debug");
  return params;
}

SecantSolver::SecantSolver(const InputParameters & parameters)
  : SplitOperatorBase(parameters),
    _substeps(getParam<unsigned int>("substeps")),
    _max_iterations(getParam<unsigned int>("max_iterations")),
    _tolerance(getParam<Real>("tolerance")),
    _du_realspace(getOutputBuffer("du_realspace"))
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

  for (const auto i : make_range(n))
  {
    auto & u_out = _variables[i]._buffer;
    const auto & u = _variables[i]._reciprocal_buffer;
    const auto & N = _variables[i]._nonlinear_reciprocal;
    const auto & L = _variables[i]._linear_reciprocal;

    // initial guess computed using semi-implicit Euler
    for (auto & cmp : _computes)
      cmp->computeBuffer();
    Rprev[i] = _domain.ifft((N + L * u) * dt); // u = u_old at this point!
    uprev[i] = u_out;

    // previous timestep solution
    if (_variables[i]._reciprocal_buffer.defined())
      u_old[i] = _variables[i]._reciprocal_buffer;
    else
      u_old[i] = _domain.fft(_variables[i]._buffer);

    // now modify u_out
    const auto dt_epsilon = getParam<Real>("dt_epsilon");
    u_out = _domain.ifft((u + dt_epsilon * N) / (1.0 - dt_epsilon * L));
  }

  // Jacobian
  torch::Tensor J;
  // Residual
  torch::Tensor R;

  // secant iterations
  for (const auto iteration : make_range(_max_iterations))
  {
    // evaluate the solve computes
    for (auto & cmp : _computes)
      cmp->computeBuffer();

    // integrate all variables
    for (const auto i : make_range(n))
    {
      auto & u_out = _variables[i]._buffer;
      const auto & u = _variables[i]._reciprocal_buffer;
      const auto & N = _variables[i]._nonlinear_reciprocal;
      const auto & L = _variables[i]._linear_reciprocal;

      // residual in real space
      R = _domain.ifft((N + L * u) * dt + u_old[i] - u);

      J = (R - Rprev[i]) / (u_out - uprev[i]);
      // J = torch::where(torch::abs(denom) > _tolerance, R - Rprev[i] / denom, 0.0);

      // std::cout << " |denom|=" << torch::sum(torch::abs(denom)).item<double>() << '\n';
      // std::cout << " |R-Rprev|=" << torch::sum(torch::abs(R - Rprev[i])).item<double>() <<
      // '\n';

      auto du = -R / J;
      // auto du = torch::where(torch::abs(J) > _tolerance, -R / J, 0.0);

      uprev[i] = u_out.clone();
      Rprev[i] = R;

      u_out += du * 0.1;

      const auto Jnorm = torch::sum(torch::abs(J)).item<double>();
      const auto Jmin = torch::min(torch::abs(J)).item<double>();
      const auto unorm = torch::sum(torch::abs(du)).item<double>();
      const auto Rnorm = torch::sum(torch::abs(R)).item<double>();
      std::cout << iteration << " |du| = " << unorm << " |R|=" << Rnorm << " |J|=" << Jnorm
                << " min(|J|)=" << Jmin << std::endl;
      if (Rnorm < _tolerance) // TODO: must apply to all variables
      {
        // _console << "Secant solve converged after " << iteration << " iterations.\n";
        _du_realspace = R;

        return;
      }
    }
  }
  _du_realspace = R;
}
