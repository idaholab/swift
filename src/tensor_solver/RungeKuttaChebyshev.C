/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2025 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "RungeKuttaChebyshev.h"
#include "TensorProblem.h"
#include "DomainAction.h"
#include <vector>
#include <algorithm>

registerMooseObject("SwiftApp", RungeKuttaChebyshev);

InputParameters
RungeKuttaChebyshev::validParams()
{
  InputParameters params = ExplicitSolverBase::validParams();
  params.addClassDescription("Runge–Kutta–Chebyshev (RKC) explicit stabilized time integrator.");
  params.addParam<unsigned int>("substeps", 1, "Explicit substeps per time step.");
  params.addRangeCheckedParam<unsigned int>(
      "stages", 1, "stages >= 1", "Number of RKC stages per substep.");
  params.addRangeCheckedParam<Real>("damping",
                                    0.0,
                                    "damping >= 0",
                                    "Damping parameter epsilon for RKC stabilization.");
  return params;
}

RungeKuttaChebyshev::RungeKuttaChebyshev(const InputParameters & parameters)
  : ExplicitSolverBase(parameters),
    _substeps(getParam<unsigned int>("substeps")),
    _stages(getParam<unsigned int>("stages")),
    _damping(getParam<Real>("damping")),
    _sub_dt(_tensor_problem.subDt()),
    _sub_time(_tensor_problem.subTime())
{
}

void
RungeKuttaChebyshev::computeBuffer()
{
  torch::Tensor ubar;
  _sub_dt = _dt / _substeps;

  // subcycles
  for (const auto substep : make_range(_substeps))
  {
    // Evaluate RHS at current state (stage 0)
    _compute->computeBuffer();
    forwardBuffers();

    if (_stages == 1)
    {
      // Single-stage: Forward Euler
      for (auto & [u, reciprocal_buffer, time_derivative_reciprocal] : _variables)
      {
        ubar = reciprocal_buffer + _sub_dt * time_derivative_reciprocal;
        u = _domain.ifft(ubar);
      }
    }
    else
    {
      // ROCK2 stabilized explicit scheme with fixed stages
      const unsigned int s = _stages;

      // Future extension note: If an estimated spectral radius L is available,
      // the number of stages s could be adapted as s ≈ 1 + sqrt(L * dt / (2 * (1 + eps))),
      // and ω0 adjusted accordingly. Here we keep s fixed and use user-provided damping.

      const Real eps = _damping;
      const Real omega0 = 1.0 + eps / (static_cast<Real>(s) * static_cast<Real>(s));

      // Chebyshev polynomials at omega0
      std::vector<Real> T(s + 1, 0.0);
      std::vector<Real> U(std::max<int>(s, 1), 0.0);
      T[0] = 1.0;
      T[1] = omega0;
      U[0] = 1.0;
      if (s > 1)
        U[1] = 2.0 * omega0;
      for (unsigned int j = 2; j <= s; ++j)
        T[j] = 2.0 * omega0 * T[j - 1] - T[j - 2];
      for (unsigned int j = 2; j < s; ++j)
        U[j] = 2.0 * omega0 * U[j - 1] - U[j - 2];

      // ω1 = T_s(ω0) / (s * U_{s-1}(ω0))
      const Real omega1 = T[s] / (static_cast<Real>(s) * U[s - 1]);

      // Stage 0 data (Y0 and F0)
      const std::size_t nvar = _variables.size();
      std::vector<torch::Tensor> Y0(nvar);
      std::vector<torch::Tensor> F0(nvar);
      for (const auto k : make_range(nvar))
      {
        auto & reciprocal_buffer = _variables[k]._reciprocal_buffer;
        auto & time_derivative_reciprocal = _variables[k]._time_derivative_reciprocal;
        Y0[k] = reciprocal_buffer.clone();
        F0[k] = time_derivative_reciprocal.clone();
      }

      // Stage 1: Y1 = Y0 + b1 * dt * F0, with b1 = ω1/ω0
      const Real b1 = omega1 / omega0;
      std::vector<torch::Tensor> Yjm2 = Y0; // Y_{j-2}
      std::vector<torch::Tensor> Yjm1(nvar); // Y_{j-1}
      std::vector<torch::Tensor> Fjm2 = F0; // F(Y_{j-2})
      std::vector<torch::Tensor> Fjm1(nvar); // F(Y_{j-1})

      for (const auto k : make_range(nvar))
      {
        auto & u = _variables[k]._buffer;
        Yjm1[k] = Y0[k] + (_sub_dt * b1) * F0[k];
        u = _domain.ifft(Yjm1[k]);
      }

      // Compute F1 = F(Y1)
      _compute->computeBuffer();
      forwardBuffers();
      for (const auto k : make_range(nvar))
        Fjm1[k] = _variables[k]._time_derivative_reciprocal.clone();

      // Stages j = 2..s
      for (unsigned int j = 2; j <= s; ++j)
      {
        // μ_j and ν_j per Abdulle ROCK2 formulation (using Chebyshev polynomials)
        const Real mu_j = 2.0 * omega1 * T[j - 1] / T[j];
        const Real nu_j = 2.0 * omega1 * (j >= 2 ? U[j - 2] : 0.0) / T[j];

        // We use the agreed combination form:
        //  a_j = 1 - μ_j ; c_j = ν_j
        //  Yj = (1 - a_j - c_j)·Y0 + a_j·Y_{j-1} + c_j·Y_{j-2}
        //       + μ_j·dt·F(Y_{j-1}) + ν_j·dt·F(Y_{j-2})
        const Real a_j = 1.0 - mu_j;
        const Real c_j = nu_j;
        const Real coeff_Y0 = 1.0 - a_j - c_j; // = μ_j - ν_j

        // Build Yj for all variables and set real buffers for stage evaluation
        std::vector<torch::Tensor> Yj(nvar);
        for (const auto k : make_range(nvar))
        {
          auto & u = _variables[k]._buffer;
          Yj[k] = coeff_Y0 * Y0[k] + a_j * Yjm1[k] + c_j * Yjm2[k] + (_sub_dt * mu_j) * Fjm1[k] +
                  (_sub_dt * nu_j) * Fjm2[k];
          u = _domain.ifft(Yj[k]);
        }

        // If this is the last stage, we are done setting u = ifft(Ys)
        if (j == s)
          break;

        // Otherwise, evaluate F(Yj) and rotate stage storage
        _compute->computeBuffer();
        forwardBuffers();
        // rotate storage: (jm2 <- jm1), (jm1 <- j)
        for (const auto k : make_range(nvar))
        {
          Yjm2[k] = Yjm1[k];
          Fjm2[k] = Fjm1[k];
          Yjm1[k] = Yj[k];
          Fjm1[k] = _variables[k]._time_derivative_reciprocal.clone();
        }
      }
    }

    if (substep < _substeps - 1)
      _tensor_problem.advanceState();

    // increment substep time
    _sub_time += _sub_dt;
  }
}
