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
  const auto dt = _dt / _substeps;

  // Jacobian dimensions
  const auto n = _variables.size();
  const auto & s = _domain.getReciprocalShape();
  std::vector<int64_t> v(s.begin(), s.end());
  v.push_back(n);
  v.push_back(n);

  // Create a 3x3 identity matrix and expand to all grid points
  torch::Tensor M = torch::eye(n, _options); // * 1e-6;
  for (const auto i : make_range(_dim))
    M.unsqueeze_(0);
  M = M.expand(v);

  // stack u_old
  std::vector<torch::Tensor> u_old_v(n);
  for (const auto i : make_range(n))
    if (_variables[i]._reciprocal_buffer.defined())
      u_old_v[i] = _variables[i]._reciprocal_buffer;
    else
      u_old_v[i] = _domain.fft(_variables[i]._buffer);
  const auto u_old = torch::stack(u_old_v, -1);

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

  // initial residual
  for (auto & cmp : _computes)
    cmp->computeBuffer();
  const auto [u0, N, L] = stackVariables();
  torch::Tensor u = u0;
  torch::Tensor R = (N + L * u) * dt;

  // initial residual norm
  const auto R0norm = torch::norm(R).item<double>();

  // secant iterations
  for (const auto iteration : make_range(_max_iterations))
  {
    // check for convergence
    const auto Rnorm = torch::norm(R).item<double>();
    if (Rnorm / R0norm < _tolerance)
    {
      std::cout << "Broyden solve converged after " << iteration << " iterations.\n";
      return;
    }
    else
      std::cout << iteration << " |R|=" << Rnorm << std::endl;

    // update step dx
    const auto sk = -torch::matmul(M, R.unsqueeze(-1)); // column vector
    const auto skT = sk.squeeze(-1).unsqueeze(-2);      // row vector

    // update u
    const auto u_out_v = torch::unbind(u + sk.squeeze(-1), -1);
    for (const auto i : make_range(n))
      _variables[i]._buffer = _domain.ifft(u_out_v[i]);
    // update residual
    for (auto & cmp : _computes)
      cmp->computeBuffer();
    const auto [u0, N, L] = stackVariables();
    u = u0;
    const auto Rnew = (N + L * u) * dt + u_old - u;

    // residual change
    const auto yk = (Rnew - R).unsqueeze(-1);
    const auto ykT = yk.squeeze(-1).unsqueeze(-2);

    const auto denom = torch::matmul(skT, yk);
    // M = M + torch::where(torch::abs(denom) > 0, torch::matmul((sk - torch::matmul(M, yk)), skT) / denom, 0.0);
    M = M + torch::matmul((sk - torch::matmul(M, yk)), skT) /
                torch::where(torch::abs(denom) > 1e-12, denom, 1.0);

    // M = M + torch::matmul((sk - torch::matmul(M, yk)), skT) / torch::matmul(skT, yk);

    R = Rnew;
  }

  std::cerr << "Broyden solve did not converge within the maximum number of iterations.\n";
}
