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
 * 2-dimensional 9 velocity lattice configuration
 */

class LBMD2Q9 : public LBMStencilBase
{
public:
  static InputParameters validParams();

  LBMD2Q9(const InputParameters & parameters);
};
