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

  // previous timestep solution
  for (const auto i : make_range(n))
  {
    if (_variables[i]._reciprocal_buffer.defined())
      u_old[i] = _variables[i]._reciprocal_buffer;
    else
      u_old[i] = _domain.fft(_variables[i]._buffer);
  }

  // inverse of the Jacobian
  torch::Tensor iJ;

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

      // residual
      auto R = (N + L * u) * dt + u_old[i] - u;

      // approximate Jacobian during first substep
      if (iteration == 0)
      {
        iJ = 1.0 / (L * dt - 1.0);
        // iJ = torch::where(torch::abs(iJ) > 1.0, 1e-5, iJ);
      }
      else
      {
        const auto denom = R - Rprev[i];
        iJ = torch::where(torch::abs(denom) > _tolerance, (u - uprev[i]) / denom, _tolerance);
      }

      auto du = -R * iJ;
      du = torch::where(torch::abs(du) > 1, du / torch::abs(du), du);

      // debug
      if (_substep == 0)
        _du_realspace = _domain.ifft(du);

      const auto unext = u + du;
      u_out = _domain.ifft(unext);

      const auto unorm = torch::sum(torch::abs(du)).item<double>();
      const auto Rnorm = torch::sum(torch::abs(R)).item<double>();
      std::cout << iteration << " |du| = " << unorm << " |R|=" << Rnorm << std::endl;
      if (unorm < _tolerance) // TODO: must apply to all variables
      {
        // _console << "Secant solve converged after " << iteration << " iterations.\n";
        return;
      }

      uprev[i] = unext;
      Rprev[i] = R;
    }
  }
}
