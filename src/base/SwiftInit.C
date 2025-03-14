/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "SwiftInit.h"
#include "SwiftApp.h"

SwiftInit::SwiftInit(int argc, char * argv[], MPI_Comm COMM_WORLD_IN)
  : MooseInit(argc, argv, COMM_WORLD_IN)
{
  for (const auto i : make_range(argc - 1))
    if (std::string(argv[i]) == "--libtorch-device")
      SwiftApp::setTorchDeviceStatic(argv[i + 1], {});
}
