/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LatticeBoltzmannStencilBase.h"
#include "LatticeBoltzmannOperator.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannMesh.h"

/**
 * Compute LB equilibrium distribution
 */
class LBMEquilibrium : public LatticeBoltzmannOperator
{
public:
    static InputParameters validParams();

    LBMEquilibrium(const InputParameters & parameters);

    virtual void computeBuffer() override;

protected:
  const torch::Tensor & _rho;
  const torch::Tensor & _velocity;
  const unsigned int & _dim;
  std::vector<int64_t> _shape;
};

