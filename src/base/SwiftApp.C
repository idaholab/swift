#include "SwiftApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "ModulesApp.h"
#include "MooseSyntax.h"
#include "SwiftUtils.h"

#include <cstdlib>

namespace MooseFFT
{
static struct SwiftGlobalSettings
{
  SwiftGlobalSettings()
  {
    if (std::getenv("SWIFT_FORCE_CPU"))
      _force_cpu = true;
  }
  bool _force_cpu;
} swift_global_settings;

bool
forceCPU()
{
  return swift_global_settings._force_cpu;
}
}

InputParameters
SwiftApp::validParams()
{
  InputParameters params = MooseApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  params.addCommandLineParam<bool>("force_cpu",
                                   "--force-cpu",
                                   false,
                                   "Use the CPU for spectral solves, even if a GPU is available.");

  return params;
}

SwiftApp::SwiftApp(InputParameters parameters) : MooseApp(parameters)
{
  SwiftApp::registerAll(_factory, _action_factory, _syntax);
  if (getParam<bool>("force_cpu"))
    MooseFFT::swift_global_settings._force_cpu = true;
}

SwiftApp::~SwiftApp() {}

void
SwiftApp::registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  ModulesApp::registerAllObjects<SwiftApp>(f, af, s);
  Registry::registerObjectsTo(f, {"SwiftApp"});
  Registry::registerActionsTo(af, {"SwiftApp"});

  /* register custom execute flags, action syntax, etc. here */
}

void
SwiftApp::registerApps()
{
  registerApp(SwiftApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
extern "C" void
SwiftApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  SwiftApp::registerAll(f, af, s);
}
extern "C" void
SwiftApp__registerApps()
{
  SwiftApp::registerApps();
}
