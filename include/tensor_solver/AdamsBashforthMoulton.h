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
 * Adams-Bashforth-Moulton semi-implicit/explicit solver
 */
class AdamsBashforthMoulton : public SplitOperatorBase
{
public:
  static InputParameters validParams();

  AdamsBashforthMoulton(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  unsigned int _substeps;
  std::size_t _predictor_order;
  std::size_t _corrector_order;
  std::size_t _corrector_steps;
  Real & _sub_dt;
  Real & _sub_time;
};
