//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html
#include "SwiftTestApp.h"
#include "SwiftApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "MooseSyntax.h"

InputParameters
SwiftTestApp::validParams()
{
  InputParameters params = SwiftApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  return params;
}

SwiftTestApp::SwiftTestApp(InputParameters parameters) : MooseApp(parameters)
{
  SwiftTestApp::registerAll(
      _factory, _action_factory, _syntax, getParam<bool>("allow_test_objects"));
}

SwiftTestApp::~SwiftTestApp() {}

void
SwiftTestApp::registerAll(Factory & f, ActionFactory & af, Syntax & s, bool use_test_objs)
{
  SwiftApp::registerAll(f, af, s);
  if (use_test_objs)
  {
    Registry::registerObjectsTo(f, {"SwiftTestApp"});
    Registry::registerActionsTo(af, {"SwiftTestApp"});
  }
}

void
SwiftTestApp::registerApps()
{
  registerApp(SwiftApp);
  registerApp(SwiftTestApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
// External entry point for dynamic application loading
extern "C" void
SwiftTestApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  SwiftTestApp::registerAll(f, af, s);
}
extern "C" void
SwiftTestApp__registerApps()
{
  SwiftTestApp::registerApps();
}
