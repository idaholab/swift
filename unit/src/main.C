/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "SwiftApp.h"
#include "gtest/gtest.h"

// Moose includes
#include "Moose.h"
#include "SwiftInit.h"
#include "AppFactory.h"

#include <fstream>
#include <string>

GTEST_API_ int
main(int argc, char ** argv)
{
  // gtest removes (only) its args from argc and argv - so this  must be before moose init
  testing::InitGoogleTest(&argc, argv);

  SwiftInit init(argc, argv);
  registerApp(SwiftApp);
  Moose::_throw_on_error = true;
  Moose::_throw_on_warning = true;

  return RUN_ALL_TESTS();
}
