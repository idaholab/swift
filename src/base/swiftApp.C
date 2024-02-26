#include "swiftApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "ModulesApp.h"
#include "MooseSyntax.h"

InputParameters
swiftApp::validParams()
{
  InputParameters params = MooseApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  return params;
}

swiftApp::swiftApp(InputParameters parameters) : MooseApp(parameters)
{
  swiftApp::registerAll(_factory, _action_factory, _syntax);
}

swiftApp::~swiftApp() {}

void 
swiftApp::registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  ModulesApp::registerAllObjects<swiftApp>(f, af, s);
  Registry::registerObjectsTo(f, {"swiftApp"});
  Registry::registerActionsTo(af, {"swiftApp"});

  /* register custom execute flags, action syntax, etc. here */
}

void
swiftApp::registerApps()
{
  registerApp(swiftApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
extern "C" void
swiftApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  swiftApp::registerAll(f, af, s);
}
extern "C" void
swiftApp__registerApps()
{
  swiftApp::registerApps();
}
