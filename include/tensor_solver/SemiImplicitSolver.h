/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "SplitOperatorBase.h"

/**
 * SemiImplicitSolver object
 */
class SemiImplicitSolver : public SplitOperatorBase
{
public:
  static InputParameters validParams();

  SemiImplicitSolver(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  unsigned int _substeps;
  Real & _sub_dt;
  Real & _sub_time;
};
