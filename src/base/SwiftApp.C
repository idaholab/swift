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

namespace MooseTensor
{
static struct SwiftGlobalSettings
{
  SwiftGlobalSettings()
  {
    const auto env = std::getenv("SWIFT_TORCH_DEVICE");
    if (env)
      _torch_device = std::string(env);
    else
      _torch_device = "";
  }
  std::string _torch_device;
} swift_global_settings;

std::string
torchDevice()
{
  return swift_global_settings._torch_device;
}
}

InputParameters
SwiftApp::validParams()
{
  InputParameters params = MooseApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  params.set<bool>("use_legacy_initial_residual_evaluation_behavior") = false;

  params.addCommandLineParam<std::string>(
      "torch_device", "--torch_device", "", "Device to use for spectral solves.");

  return params;
}

SwiftApp::SwiftApp(InputParameters parameters) : MooseApp(parameters)
{
  SwiftApp::registerAll(_factory, _action_factory, _syntax);
  MooseTensor::swift_global_settings._torch_device = parameters.get<std::string>("torch_device");
}

SwiftApp::~SwiftApp() {}

void
SwiftApp::registerAll(Factory & f, ActionFactory & af, Syntax & syntax)
{
  ModulesApp::registerAllObjects<SwiftApp>(f, af, syntax);
  Registry::registerObjectsTo(f, {"SwiftApp"});
  Registry::registerActionsTo(af, {"SwiftApp"});

  // FFTBuffer Actions
  registerSyntaxTask("AddTensorBufferAction", "FFTBuffers/*", "add_fft_buffer");
  syntax.registerSyntaxType("FFTBuffers/*", "FFTInputBufferName");
  syntax.registerSyntaxType("FFTBuffers/*", "FFTOutputBufferName");
  registerMooseObjectTask("add_fft_buffer", FFTBuffer, false);
  addTaskDependency("add_fft_buffer", "add_aux_variable");

  // TensorOperator Actions
  registerSyntaxTask("AddTensorObjectAction", "FFTComputes/*", "add_tensor_compute");
  syntax.registerSyntaxType("FFTComputes/*", "FFTComputeName");
  registerMooseObjectTask("add_tensor_compute", TensorOperator, false);
  addTaskDependency("add_tensor_compute", "add_fft_buffer");

  // TensorICs Actions
  registerSyntaxTask("AddTensorObjectAction", "TensorICs/*", "add_tensor_ic");
  syntax.registerSyntaxType("TensorICs/*", "FFTICName");
  registerMooseObjectTask("add_tensor_ic", TensorInitialCondition, false);
  addTaskDependency("add_tensor_ic", "add_tensor_compute");

  // TensorICs Actions
  registerSyntaxTask("AddTensorObjectAction", "FFTTimeIntegrators/*", "add_tensor_time_integrator");
  syntax.registerSyntaxType("FFTTimeIntegrators/*", "FFTTimeIntegratorName");
  registerMooseObjectTask("add_tensor_time_integrator", TensorTimeIntegrator, false);
  addTaskDependency("add_tensor_time_integrator", "add_tensor_ic");

  // FFTOutputs Actions
  registerSyntaxTask("AddTensorObjectAction", "FFTOutputs/*", "add_tensor_output");
  syntax.registerSyntaxType("FFTOutputs/*", "FFTOutputName");
  registerMooseObjectTask("add_tensor_output", TensorOutput, false);
  addTaskDependency("add_tensor_output", "add_tensor_time_integrator");

  // make sure all this gets run before `add_mortar_variable`
  addTaskDependency("add_mortar_variable", "add_tensor_output");
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
