/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "torch/torch.h"

#include "MooseObject.h"
#include "SwiftTypes.h"
#include "SwiftUtils.h"

/**
 * Base class for lattice stencils
 */

class LatticeBoltzmannStencilBase : public MooseObject
{
public:
  static InputParameters validParams();

  LatticeBoltzmannStencilBase(const InputParameters & parameters);

public:
  int64_t _q;
  torch::Tensor _ex;
  torch::Tensor _ey;
  torch::Tensor _ez;
  torch::Tensor _weights;
  torch::Tensor _M;
  torch::Tensor _M_inv;
  torch::Tensor _S;
  
  // incoming unknown distribution functions at every face
  torch::Tensor _top;
  torch::Tensor _bottom;
  torch::Tensor _left;
  torch::Tensor _right;
  torch::Tensor _front;
  torch::Tensor _back;

  // indices of opposite directions
  torch::Tensor _op;
};
