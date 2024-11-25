/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MooseObject.h"
#include "SwiftTypes.h"

class TensorBufferBase : public MooseObject
{
public:
  static InputParameters validParams();

  TensorBufferBase(const InputParameters & parameters);
};
