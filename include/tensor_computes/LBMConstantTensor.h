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
 * LBMConstantTensor object
 */
class LBMConstantTensor : public LatticeBoltzmannOperator
{
public:
  static InputParameters validParams();

  LBMConstantTensor(const InputParameters & parameters);

  virtual void computeBuffer() override;
};
