/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "LatticeBoltzmannStencilBase.h"

/**
 * 3-dimensional 27 velocity lattice configuration
 */

class LBMD3Q27 : public LatticeBoltzmannStencilBase
{
public:
  static InputParameters validParams();

  LBMD3Q27(const InputParameters & parameters);
};
