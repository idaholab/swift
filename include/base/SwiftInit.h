/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MooseInit.h"

class SwiftInit : public MooseInit
{
public:
  SwiftInit(int argc, char * argv[], MPI_Comm COMM_WORLD_IN = MPI_COMM_WORLD);
};
