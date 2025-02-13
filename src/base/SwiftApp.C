/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "SwiftApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "ModulesApp.h"
#include "MooseSyntax.h"
#include "DomainAction.h"
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
  return params;
}

SwiftApp::SwiftApp(InputParameters parameters) : MooseApp(parameters)
{
  SwiftApp::registerAll(_factory, _action_factory, _syntax);
  MooseTensor::swift_global_settings._torch_device =
      std::string(parameters.get<MooseEnum>("libtorch_device"));
}

SwiftApp::~SwiftApp() {}

void
SwiftApp::setTorchDevice(std::string device, const MooseTensor::Key<DomainAction> &)
{
  MooseTensor::swift_global_settings._torch_device = device;
}

void
SwiftApp::setTorchDeviceStatic(std::string device, const MooseTensor::Key<SwiftInit> &)
{
  MooseTensor::swift_global_settings._torch_device = device;
}

void
SwiftApp::registerAll(Factory & f, ActionFactory & af, Syntax & syntax)
{
  ModulesApp::registerAllObjects<SwiftApp>(f, af, syntax);
  Registry::registerObjectsTo(f, {"SwiftApp"});
  Registry::registerActionsTo(af, {"SwiftApp"});

  // ComputeDevice Action
  registerSyntax("DomainAction", "Domain");

  // TensorBuffer Actions
  registerSyntaxTask("AddTensorBufferAction", "TensorBuffers/*", "add_tensor_buffer");
  syntax.registerSyntaxType("TensorBuffers/*", "TensorInputBufferName");
  syntax.registerSyntaxType("TensorBuffers/*", "TensorOutputBufferName");
  registerMooseObjectTask("add_tensor_buffer", TensorBuffer, false);
  addTaskDependency("add_tensor_buffer", "add_aux_variable");

  // TensorComputes/Initial Actions
  registerSyntaxTask("AddTensorObjectAction", "TensorComputes/Initialize/*", "add_tensor_ic");
  syntax.registerSyntaxType("TensorComputes/Initialize/*", "TensorComputeName");
  registerMooseObjectTask("add_tensor_ic", TensorOperator, false);
  addTaskDependency("add_tensor_ic", "add_tensor_buffer");

  // TensorComputes/Solve Action
  registerSyntaxTask("AddTensorObjectAction", "TensorComputes/Solve/*", "add_tensor_compute");
  syntax.registerSyntaxType("TensorComputes/Solve/*", "TensorComputeName");
  registerMooseObjectTask("add_tensor_compute", TensorOperator, false);
  addTaskDependency("add_tensor_compute", "add_tensor_buffer");

  // TensorComputes/OnDemand Action
  registerSyntaxTask("AddTensorObjectAction", "TensorComputes/OnDemand/*", "add_tensor_on_demand");
  syntax.registerSyntaxType("TensorComputes/OnDemand/*", "TensorComputeName");
  registerMooseObjectTask("add_tensor_on_demand", TensorOperator, false);
  addTaskDependency("add_tensor_on_demand", "add_tensor_buffer");

  // TensorComputes/Postprocess Action
  registerSyntaxTask(
      "AddTensorObjectAction", "TensorComputes/Postprocess/*", "add_tensor_postprocessor");
  syntax.registerSyntaxType("TensorComputes/Postprocess/*", "TensorComputeName");
  registerMooseObjectTask("add_tensor_postprocessor", TensorOperator, false);
  addTaskDependency("add_tensor_postprocessor", "add_tensor_compute");

  registerSyntaxTask("EmptyAction", "TensorComputes", "no_action"); // placeholder

  // TensorTimeIntegrator Action
  registerSyntaxTask(
      "AddTensorObjectAction", "TensorTimeIntegrators/*", "add_tensor_time_integrator");
  syntax.registerSyntaxType("TensorTimeIntegrators/*", "TensorTimeIntegratorName");
  registerMooseObjectTask("add_tensor_time_integrator", TensorTimeIntegrator, false);
  addTaskDependency("add_tensor_time_integrator", "add_tensor_postprocessor");

  // TensorOutputs Action
  registerSyntaxTask("AddTensorObjectAction", "TensorOutputs/*", "add_tensor_output");
  syntax.registerSyntaxType("TensorOutputs/*", "TensorOutputName");
  registerMooseObjectTask("add_tensor_output", TensorOutput, false);
  addTaskDependency("add_tensor_output", "add_tensor_time_integrator");

  // Create TensorSolver
  registerSyntaxTask("CreateTensorSolverAction", "TensorSolver", "create_tensor_solver");
  registerMooseObjectTask("create_tensor_solver", TensorSolver, false);
  addTaskDependency("create_tensor_solver", "add_tensor_output");

  // Add predictors to the solver
  registerSyntaxTask(
      "AddTensorPredictorAction", "TensorSolver/Predictors/*", "add_tensor_predictor");
  syntax.registerSyntaxType("TensorSolver/Predictors/*", "TensorPredictorName");
  registerMooseObjectTask("add_tensor_predictor", TensorPredictor, false);
  addTaskDependency("add_tensor_predictor", "create_tensor_solver");

  // make sure all this gets run before `add_mortar_variable`
  addTaskDependency("add_mortar_variable", "add_tensor_predictor");
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
