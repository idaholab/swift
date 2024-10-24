//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "BroydenSolver.h"
#include "TensorProblem.h"
#include "DomainAction.h"

registerMooseObject("SwiftApp", BroydenSolver);

InputParameters
BroydenSolver::validParams()
{
  InputParameters params = SplitOperatorBase::validParams();
  params.addClassDescription("Implicit secant solver time integration.");
  params.addParam<unsigned int>("substeps", 1, "secant solver substeps per time step.");
  params.addParam<unsigned int>("max_iterations", 5, "Maximum number of secant solver iteration.");
  params.addParam<Real>("tolerance", 1e-10, "Convergence tolerance.");
  params.addParam<Real>("damping", 1.0, "Damping factor for teh update step.");
  params.addParam<Real>(
      "dt_epsilon", 1e-4, "Semi-implicit stable timestep to bootstrap secant solve.");
  params.set<unsigned int>("substeps") = 0;
  params.addParam<bool>("verbose", false, "Show convergence history.");
  return params;
}

BroydenSolver::BroydenSolver(const InputParameters & parameters)
  : SplitOperatorBase(parameters),
    _substeps(getParam<unsigned int>("substeps")),
    _max_iterations(getParam<unsigned int>("max_iterations")),
    _tolerance(getParam<Real>("tolerance")),
    _verbose(getParam<bool>("verbose")),
    _damping(getParam<Real>("damping")),
    _dim(_domain.getDim()),
    _options(MooseTensor::complexFloatTensorOptions())
{
}

void
BroydenSolver::computeBuffer()
{
  for (_substep = 0; _substep < _substeps; ++_substep)
    broydenSolve();
}

void
BroydenSolver::broydenSolve()
{
  // Jacobian dimensions
  const auto n = _variables.size();
  const auto & s = _domain.getReciprocalShape();
  std::vector<int64_t> v(s.begin(), s.end());
  v.push_back(n);
  v.push_back(n);

  // Create a 3x3 identity matrix and expand to all grid points
  torch::Tensor M = torch::eye(n, _options);
  for (const auto i : make_range(_dim))
    M.unsqueeze_(0);
  M = M.expand(v);

  auto stackVariables = [&]()
  {
    std::vector<torch::Tensor> u(n);
    std::vector<torch::Tensor> N(n);
    std::vector<torch::Tensor> L(n);
    for (const auto i : make_range(n))
    {
      u[i] = _variables[i]._reciprocal_buffer;
      N[i] = _variables[i]._nonlinear_reciprocal;
      L[i] = _variables[i]._linear_reciprocal;
    }
    return std::make_tuple(torch::stack(u, -1), torch::stack(N, -1), torch::stack(L, -1));
  };

  for (auto & cmp : _computes)
    cmp->computeBuffer();

  const auto [u, N, L] = stackVariables();
  const auto dt = _dt / _substeps;

  // initial residual
  const auto R0 = (N + L * u) * dt;
  torch::Tensor R = R0;

  // stack u_old
  std::vector<torch::Tensor> u_old_v(n);
  for (const auto i : make_range(n))
    if (_variables[i]._reciprocal_buffer.defined())
      u_old_v[i] = _variables[i]._reciprocal_buffer;
    else
      u_old_v[i] = _domain.fft(_variables[i]._buffer);
  const auto u_old = torch::stack(u_old_v, -1);

  // secant iterations
  bool all_converged;
  for (const auto iteration : make_range(_max_iterations))
  {
    const auto sv = torch::matmul(M, R.unsqueeze(-1)).squeeze(-1);

    const auto s = sv.unsqueeze(-2);
    const auto sT = sv.unsqueeze(-1);

    // update u
    const auto u_out_v = torch::unbind(u + sv, -1);
    for (const auto i : make_range(n))
      _variables[i]._buffer = _domain.ifft(u_out_v[i]);

    // evaluate the solve computes
    for (auto & cmp : _computes)
      cmp->computeBuffer();

    const auto [u, N, L] = stackVariables();

    // residual in reciprocal space
    R = (N + L * u) * dt + u_old - u;

    MooseTensor::printTensorInfo(torch::matmul(s, sT));
    mooseError("done");

    // Vector fx_new = f(x + s);
    // Vector y = fx_new - fx;
    // Vector dy = fx_new - fx_new.subvec(0, n - 2); // Approximate the Jacobian column
    // M = M + (y * dy.t()) / (dy.t() * dy) - (s * s.t()) / (s.t() * y);
    // x += s;
    // fx = fx_new;

    // if (arma::norm(fx, 2) < epsilon) {
    //   return x; // Converged
    // }

    // // avoid NaN
    // const auto denom = u - uprev[i];
    // J = (R - Rprev[i]) / denom;
    // auto du = torch::where(torch::abs(denom) > 0, -R / J, 0.0);

    // uprev[i] = u;
    // Rprev[i] = R;

    // if (_damping == 1.0)
    //   u_out = _domain.ifft(u + du);
    // else
    //   u_out = _domain.ifft(u + du * _damping);

    // const auto Rnorm = torch::sum(torch::abs(R)).item<double>();

    // if (_verbose)
    // {
    //   const auto unorm = torch::sum(torch::abs(du)).item<double>();
    //   std::cout << iteration << " |du| = " << unorm << " |R|=" << Rnorm << std::endl;
    // }

    // // relative convergence check
    // all_converged = all_converged && (Rnorm / R0norm[i] < _tolerance);

    // if (all_converged)
    // {
    //   std::cout << "Secant solve converged after " << iteration << " iterations.\n";
    //   break;
    // }
  }

  if (!all_converged)
    std::cout << "Solve not converged.\n";
}
