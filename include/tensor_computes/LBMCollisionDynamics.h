  /**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LatticeBoltzmannOperator.h"

/**
  * Template object for LBM collision dynamics
  */
template<int coll_dyn>
class LBMCollisionDynamicsTempl: public LatticeBoltzmannOperator
{
public: 
  static InputParameters validParams();

  LBMCollisionDynamicsTempl(const InputParameters & parameters);

  void BGKDynamics();
  void MRTDynamics();
  void SmagorinskyDynamics();

  void HermiteRegularization();

  void computeBuffer() override;

protected:
  const torch::Tensor & _f;
  const torch::Tensor & _feq;
  torch::Tensor _fneq;
  const std::array<int64_t, 3> _shape;

  const Real _tau_0;
  const Real _C_s; // Smagorinsky constant
  const Real _mean_density;
  const bool _projection;
};

typedef LBMCollisionDynamicsTempl<0> LBMBGKCollision;
typedef LBMCollisionDynamicsTempl<1> LBMMRTCollision;
typedef LBMCollisionDynamicsTempl<2> LBMSmagorinskyCollision;
