/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LBMStencilBase.h"

/**
 * 3-dimensional 19 velocity lattice configuration
 */

class LBMD3Q19 : public LBMStencilBase
{
public:
  static InputParameters validParams();

  LBMD3Q19(const InputParameters & parameters);
};
