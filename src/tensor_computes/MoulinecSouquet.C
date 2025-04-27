/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "MoulinecSouquet.h"
#include "DomainAction.h"
#include "SwiftUtils.h"
#include <ATen/TensorIndexing.h>
#include <torch/linalg.h>
#ifdef NEML2_ENABLED

registerMooseObject("SwiftApp", MoulinecSouquet);

InputParameters
MoulinecSouquet::validParams()
{
  InputParameters params = TensorOperator<neml2::SR2>::validParams();
  params.addClassDescription("Moulinec-Souquet mechanics solve.");
  params.addParam<TensorInputBufferName>("C0", "Stiffness tensor estimate");
  return params;
}

MoulinecSouquet::MoulinecSouquet(const InputParameters & parameters)
  : TensorOperator<neml2::SR2>(parameters),
    // _C0(getInputBuffer<neml2::SSR4>("C0"))
    _C0(neml2::SSR4::isotropic_E_nu(100, 0.3, MooseTensor::floatTensorOptions())),
    _kvec({&_i, &_j, &_k}),
    _map({{{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}}}),
    _f({1.0, 1.0, 1.0, std::sqrt(2.0), std::sqrt(2.0), std::sqrt(2.0)}),
    _inv_f({1.0, 1.0, 1.0, std::sqrt(1.0 / 2.0), std::sqrt(1.0 / 2.0), std::sqrt(1.0 / 2.0)})
{
  // TODO: support dynamic C0 averages
  updateGamma();
}

void
MoulinecSouquet::computeBuffer()
{
}

void
MoulinecSouquet::updateGamma()
{
  using namespace torch::indexing;

  // 1) stack your 3 components of k together: [...,3]
  auto kstack = torch::stack({_i.expand(_domain.getReciprocalShape()), _j.expand(_domain.getReciprocalShape()), _k.expand(_domain.getReciprocalShape())}, /*dim=*/-1); // shape [...,3]

  // 2) build P and Q
  auto P = torch::zeros({3, 6}, MooseTensor::complexFloatTensorOptions());
  auto Q = torch::zeros({6, 3}, MooseTensor::complexFloatTensorOptions());
  for (int M = 0; M < 6; ++M)
  {
    int j = _map[M].second;
    P.index_put_({j, M}, _inv_f[M]);
  }
  for (int N = 0; N < 6; ++N)
  {
    int L = _map[N].first;
    Q.index_put_({N, L}, _inv_f[N]);
  }

  // 3) now
  auto v = torch::einsum("...j,jM->...iM", std::vector<at::Tensor>{kstack, P});
  //    → v[...,i,M] = sum_j k[...,j]*P[j,M] = k[...,j(M)]/f[M]

  auto w = torch::einsum("...j,Nj->...Nl", std::vector<at::Tensor>{kstack, Q});
  //    → w[...,N,L] = sum_j k[...,j]*Q[N,j] = k[...,m(N)]/f[N]

  // "...iM,MN,...Nl->...il" sums over M and N, produces [...,3,3]
  torch::Tensor Nmat;
  Nmat = torch::einsum(
      "...iM,MN,...Nl->...il",
      std::vector<at::Tensor>{v, _C0.to(MooseTensor::complexFloatTensorOptions()), w});

  const Real E = 100.0, nu = 0.3;
  const auto lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
  const auto mu = E / (2 * (1 + nu));
  auto k2 = _i * _i + _j * _j + _k * _k;
  auto k4 = k2 * k2;

  // analytic N for comparison:
  for (int i = 0; i < 3; ++i)
    for (int l = 0; l < 3; ++l)
    {
      auto N_analytic = torch::where(k2 > 0,
                                     (lambda + 2 * mu) * (*_kvec[i]) * (*_kvec[l]) +
                                         mu * k2 * (i == l ? 1.0 : 0.0),
                                     0.0);
      auto diff = (Nmat.index({Ellipsis, i, l}) - N_analytic).abs().max();
      std::cout << "max|N(" << i << "," << l << ") – analytic| = " << diff.cpu().item<double>()
                << "\n";
    }

  // invert projected stiffness
  auto Mmat = torch::linalg::pinv(Nmat);

  // allocate Gamma operator
  _gamma = _domain.emptyReciprocal({6, 6});

  // assemble via explicit M,N loops
  for (const auto M : make_range(6))
  {
    const auto [i, j] = _map[M];
    for (const auto N : make_range(6))
    {
      const auto [l, m] = _map[N];

      // four‐term symmetrized contraction, each term is [nx,ny]
      auto t1 = Mmat.index({Ellipsis, i, l}) * *_kvec[j] * *_kvec[m];
      auto t2 = Mmat.index({Ellipsis, i, m}) * *_kvec[j] * *_kvec[l];
      auto t3 = Mmat.index({Ellipsis, j, l}) * *_kvec[i] * *_kvec[m];
      auto t4 = Mmat.index({Ellipsis, j, m}) * *_kvec[i] * *_kvec[l];

      // average, re‑apply Mandel factors f[M], f[N], and store
      auto Gmn = 0.5 * (t1 + t2 + t3 + t4);
      _gamma.index_put_({Ellipsis, M, N}, _f[M] * _f[N] * Gmn);
    }
  }

  // std::cout << Mmat << '\n';

  Mmat = _domain.emptyReciprocal({3, 3});
  for (const auto i : make_range(3))
    for (const auto j : make_range(3))
    {
      Mmat.index_put_(
          {Ellipsis, i, j},
          torch::where(k2 > 0,
                       (i == j ? 1.0 / (mu * k2) : torch::zeros_like(k2)) -
                           (lambda + mu) / (mu * (lambda + 2 * mu) * k4) * *_kvec[i] * *_kvec[j],
                       0.0));
    }
  // std::cout << Mmat << '\n';
}

#endif
