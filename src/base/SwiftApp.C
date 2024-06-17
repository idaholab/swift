//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

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
  params.set<bool>("use_legacy_initial_residual_evaluation_behavior") = false;

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
SwiftApp::registerAll(Factory & f, ActionFactory & af, Syntax & syntax)
{
  ModulesApp::registerAllObjects<SwiftApp>(f, af, syntax);
  Registry::registerObjectsTo(f, {"SwiftApp"});
  Registry::registerActionsTo(af, {"SwiftApp"});

  // FFTBuffer Actions
  registerSyntaxTask("AddFFTBufferAction", "FFTBuffers/*", "add_fft_buffer");
  syntax.registerSyntaxType("FFTBuffers/*", "FFTInputBufferName");
  syntax.registerSyntaxType("FFTBuffers/*", "FFTOutputBufferName");
  registerMooseObjectTask("add_fft_buffer", FFTBuffer, false);
  addTaskDependency("add_fft_buffer", "create_problem_complete");
  addTaskDependency("add_fft_buffer", "create_problem_complete");

  // FFTCompute Actions
  registerSyntaxTask("AddFFTComputeAction", "FFTComputes/*", "add_fft_compute");
  syntax.registerSyntaxType("FFTComputes/*", "FFTComputeName");
  registerMooseObjectTask("add_fft_compute", FFTCompute, false);
  addTaskDependency("add_fft_compute", "add_fft_buffer");

  // FFTICs Actions
  registerSyntaxTask("AddFFTComputeAction", "FFTICs/*", "add_fft_ic");
  syntax.registerSyntaxType("FFTICs/*", "FFTICName");
  registerMooseObjectTask("add_fft_ic", FFTInitialCondition, false);
  addTaskDependency("add_fft_ic", "add_fft_compute");

  // make sure all this gets run before `init_physics`
  addTaskDependency("init_physics", "add_fft_ic");
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
