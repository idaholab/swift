//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html
#include "swiftTestApp.h"
#include "swiftApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "MooseSyntax.h"

InputParameters
swiftTestApp::validParams()
{
  InputParameters params = swiftApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  return params;
}

swiftTestApp::swiftTestApp(InputParameters parameters) : MooseApp(parameters)
{
  swiftTestApp::registerAll(
      _factory, _action_factory, _syntax, getParam<bool>("allow_test_objects"));
}

swiftTestApp::~swiftTestApp() {}

void
swiftTestApp::registerAll(Factory & f, ActionFactory & af, Syntax & s, bool use_test_objs)
{
  swiftApp::registerAll(f, af, s);
  if (use_test_objs)
  {
    Registry::registerObjectsTo(f, {"swiftTestApp"});
    Registry::registerActionsTo(af, {"swiftTestApp"});
  }
}

void
swiftTestApp::registerApps()
{
  registerApp(swiftApp);
  registerApp(swiftTestApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
// External entry point for dynamic application loading
extern "C" void
swiftTestApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  swiftTestApp::registerAll(f, af, s);
}
extern "C" void
swiftTestApp__registerApps()
{
  swiftTestApp::registerApps();
}
