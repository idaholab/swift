//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "TensorProblem.h"
#include "UniformTensorMesh.h"

#include "TensorOperatorBase.h"
#include "TensorTimeIntegrator.h"
#include "TensorOutput.h"
#include "DomainAction.h"

#include "SwiftUtils.h"
#include "DependencyResolverInterface.h"

registerMooseObject("SwiftApp", TensorProblem);

InputParameters
TensorProblem::validParams()
{
  InputParameters params = FEProblem::validParams();
  params.addClassDescription(
      "A normal Problem object that adds the ability to perform spectral solves.");
  params.set<bool>("skip_nl_system_check") = true;
  params.addParam<bool>("print_debug_output", false, "Show Tensor specific debug outputs");
  params.addParam<unsigned int>(
      "spectral_solve_substeps",
      1,
      "How many substeps to divide the spectral solve for each MOOSE timestep into.");
  return params;
}

TensorProblem::TensorProblem(const InputParameters & parameters)
  : FEProblem(parameters),
    DomainInterface(this),
    _options(MooseTensor::floatTensorOptions()),
    _debug(getParam<bool>("print_debug_output")),
    _substeps(getParam<unsigned int>("spectral_solve_substeps")),
    _dim(_domain.getDim()),
    _grid_spacing(_domain.getGridSpacing()),
    _n((_domain.getGridSize())),
    _shape(_domain.getShape()),
    _solver(nullptr)
{
  // make sure AuxVariables are contiguous in teh solution vector
  getAuxiliarySystem().sys().identify_variable_groups(false);
}

TensorProblem::~TensorProblem()
{
  // wait for outputs to be completed (otherwise resources might get freed that the output thread
  // depends on)
  for (auto & output : _outputs)
    output->waitForCompletion();
}

void
TensorProblem::init()
{

  // initialize tensors (assuming all scalar for now, but in the future well have an TensorBufferBase
  // pointer as well)
  for (auto pair : _tensor_buffer)
    pair.second = torch::zeros(_shape, _options);

  // compute grid dependent quantities
  gridChanged();

  // dependency resolution of TensorICs
  DependencyResolverInterface::sort(_ics);

  // dependency resolution of TensorComputes
  DependencyResolverInterface::sort(_computes);

  // dependency resolution of Tensor Postprocessors
  DependencyResolverInterface::sort(_pps);

  // show computes
  if (_debug)
  {
    _console << COLOR_CYAN << "Compute object execution order:\n" << COLOR_DEFAULT;
    for (auto & cmp : _computes)
    {
      _console << "  " << cmp->name() << '\n' << COLOR_YELLOW;
      for (const auto & ri : cmp->getRequestedItems())
        _console << "    <- " << ri << '\n';
      _console << COLOR_GREEN;
      for (const auto & si : cmp->getSuppliedItems())
        _console << "    -> " << si << '\n';
      _console << COLOR_DEFAULT;
    }
  }

  // call base class init
  FEProblem::init();

  // init outputs
  for (auto & output : _outputs)
    output->init();

  updateDOFMap();

  // debug output
  std::string variable_mapping;
  for (const auto & [buffer_name, tuple] : _buffer_to_var)
    variable_mapping += (std::get<bool>(tuple) ? "NODAL     " : "ELEMENTAL ") + buffer_name + '\n';
  if (!variable_mapping.empty())
    mooseInfo("Direct buffer to solution vector mappings:\n", variable_mapping);
}

void
TensorProblem::execute(const ExecFlagType & exec_type)
{
  if (exec_type == EXEC_INITIAL)
  {
    // run ICs
    for (auto & ic : _ics)
      ic->computeBuffer();

    // compile ist of compute output tensors
    std::set<std::string> _is_output;
    for (auto & cmp : _computes)
      _is_output.insert(cmp->getSuppliedItems().begin(), cmp->getSuppliedItems().end());

    // check for uninitialized tensors
    for (auto & [name, t] : _tensor_buffer)
      if (!t.defined() && _is_output.count(name) == 0)
        mooseWarning(name, " is not initialized and not an output of any [Solve] compute.");
  }

  if (exec_type == EXEC_TIMESTEP_BEGIN)
  {
    // legacy time integrator system
    if (!_solver)
    {
      // if the time step changed and the current time integrator does not support variable time
      // step size, we clear the histories
      if (dt() != dtOld())
        for (auto & [name, max_states] : _old_tensor_buffer)
          max_states.second.clear();

      // update substepping dt
      _sub_dt = dt() / _substeps;

      for (unsigned substep = 0; substep < _substeps; ++substep)
      {
        // run computes on begin
        for (auto & cmp : _computes)
          cmp->computeBuffer();

        // run timeintegrators
        for (auto & ti : _time_integrators)
          ti->computeBuffer();

        // advance step (this will not work with solve failures!)
        if (substep < _substeps - 1)
          advanceState();
      }
    }
    else
      // new time integrator
      _solver->computeBuffer();

    // run postprocessing before output
    for (auto & pp : _pps)
      pp->computeBuffer();

    // wait for prior asynchronous activity on CPU buffers to complete
    // (this is a synchronization barrier for the threaded CPU activity)
    for (auto & output : _outputs)
      output->waitForCompletion();

    // prepare CPU buffers (this is a synchronization barrier for the GPU)
    for (auto & [name, cpu_buffer] : _tensor_cpu_buffer)
    {
      // get main buffer (GPU or CPU) - we already verified that it must exist
      const auto & buffer = _tensor_buffer[name];
      if (buffer.is_cpu())
        cpu_buffer = buffer.clone().contiguous();
      else
        cpu_buffer = buffer.cpu().contiguous();
    }

    // run direct buffer outputs (asynchronous in threads)
    for (auto & output : _outputs)
      output->startOutput();

    mapBuffersToAux();
  }

  FEProblem::execute(exec_type);
}

void
TensorProblem::updateDOFMap()
{
  TIME_SECTION("update", 3, "Updating Tensor DOF Map", true);

  // variable mapping
  const auto & aux = getAuxiliarySystem();
  if (!const_cast<libMesh::System &>(aux.system()).is_initialized())
    mooseError("Aux system is not initialized :(");

  auto sys_num = aux.number();
  for (auto & [buffer_name, tuple] : _buffer_to_var)
  {
    auto & [var, dofs, is_nodal] = tuple;
    if (var->isArray() || var->isVector() || var->isFV())
      mooseError("Unsupported variable type for mapping");
    auto var_num = var->number();

    const static Point shift(
        _grid_spacing[0] / 2.0, _grid_spacing[1] / 2.0, _grid_spacing[2] / 2.0);
    auto compute_iteration_index = [this](Point p, long int n0, long int n1)
    {
      switch (_dim)
      {
        case 1:
          return static_cast<long int>(p(0) / _grid_spacing[0]);

        case 2:
          return static_cast<long int>(p(0) / _grid_spacing[0]) +
                 static_cast<long int>(p(1) / _grid_spacing[1]) * n0;

        case 3:
          return static_cast<long int>(p(0) / _grid_spacing[0]) +
                 static_cast<long int>(p(1) / _grid_spacing[1]) * n0 +
                 static_cast<long int>(p(2) / _grid_spacing[2]) * n0 * n1;
        default:
          mooseError("Unsupported dimension");
      }
    };

    if (is_nodal)
    {
      long int n0 = _n[0] + 1;
      long int n1 = _n[1] + 1;
      long int n2 = _n[2] + 1;

      switch (_dim)
      {
        case 1:
          dofs.resize(n0);
          break;
        case 2:
          dofs.resize(n0 * n1);
          break;
        case 3:
          dofs.resize(n0 * n1 * n2);
          break;
        default:
          mooseError("unsupported dimension");
      }

      // loop over nodes
      for (const auto & node : _mesh.getMesh().node_ptr_range())
      {
        const auto dof_index = node->dof_number(sys_num, var_num, 0);
        const auto iteration_index = compute_iteration_index(*node + shift, n0, n1);
        dofs[iteration_index] = dof_index;
      }
    }
    else
    {
      long int n0 = _n[0];
      long int n1 = _n[1];
      long int n2 = _n[2];
      dofs.resize(n0 * n1 * n2);
      // loop over elements
      for (const auto & elem : _mesh.getMesh().element_ptr_range())
      {
        const auto dof_index = elem->dof_number(sys_num, var_num, 0);
        const auto iteration_index = compute_iteration_index(elem->centroid(), n0, n1);
        dofs[iteration_index] = dof_index;
      }
    }
  }
}

void
TensorProblem::mapBuffersToAux()
{
  // nothing to map?
  if (_buffer_to_var.empty())
    return;

  TIME_SECTION("update", 3, "Mapping Tensor buffers to Variables", true);

  auto * current_solution = &getAuxiliarySystem().solution();
  auto * solution_vector = dynamic_cast<PetscVector<Number> *>(current_solution);
  if (!solution_vector)
    mooseError(
        "Cannot map directly to the solution vector because NumericVector is not a PetscVector!");

  auto value = solution_vector->get_array();

  // const monomial variables
  for (const auto & [buffer_name, tuple] : _buffer_to_var)
  {
    const auto & [var, dofs, is_nodal] = tuple;
    libmesh_ignore(var);
    const long int n0 = is_nodal ? _n[0] + 1 : _n[0];
    const long int n1 = is_nodal ? _n[1] + 1 : _n[1];
    const long int n2 = is_nodal ? _n[2] + 1 : _n[2];

    const auto buffer = _tensor_cpu_buffer[buffer_name];
    std::size_t idx = 0;
    switch (_dim)
    {
      {
        case 1:
          const auto b = buffer.accessor<double, 1>();
          for (const auto i : make_range(n0))
            value[dofs[idx++]] = b[i % _n[0]];
          break;
      }
      case 2:
      {
        const auto b = buffer.accessor<double, 2>();
        for (const auto j : make_range(n1))
          for (const auto i : make_range(n0))
            value[dofs[idx++]] = b[i % _n[0]][j % _n[1]];
        break;
      }
      case 3:
      {
        const auto b = buffer.accessor<double, 3>();
        for (const auto k : make_range(n2))
          for (const auto j : make_range(n1))
            for (const auto i : make_range(n0))
              value[dofs[idx++]] = b[i % _n[0]][j % _n[1]][k % _n[2]];
        break;
      }
      default:
        mooseError("Unsupported dimension");
    }
  }

  solution_vector->restore_array();
  getAuxiliarySystem().sys().update();
}

void
TensorProblem::advanceState()
{
  FEProblem::advanceState();

  if (timeStep() <= 1)
    return;

  // move buffers in time
  for (auto & [name, max_states] : _old_tensor_buffer)
  {
    auto & [max, states] = max_states;
    if (states.size() < max)
      states.push_back(torch::tensor({}, _options));
    if (!states.empty())
    {
      for (std::size_t i = states.size() - 1; i > 0; --i)
        states[i] = states[i - 1];
      states[0] = _tensor_buffer[name];
    }
  }
}

void
TensorProblem::gridChanged()
{
  // _domain.gridChanged();
}

void
TensorProblem::addTensorBuffer(const std::string & buffer_name, InputParameters & parameters)
{
  // add buffer
  if (_tensor_buffer.find(buffer_name) != _tensor_buffer.end())
    mooseError("TensorBuffer '", buffer_name, "' already exists in the system");
  _tensor_buffer.try_emplace(buffer_name);

  // store variable mapping
  if (parameters.isParamValid("map_to_aux_variable"))
  {
    const auto & aux = getAuxiliarySystem();
    const auto & var_name = parameters.get<AuxVariableName>("map_to_aux_variable");
    if (!aux.hasVariable(var_name))
      mooseError("AuxVariable '", var_name, "' does not exist in the system.");

    bool is_nodal;
    const auto & var = aux.getVariable(0, var_name);
    if (var.feType() == FEType(FIRST, LAGRANGE))
      is_nodal = true;
    else if (var.feType() == FEType(CONSTANT, MONOMIAL))
      is_nodal = false;
    else
      mooseError("Only first order lagrange and constant monomial variables are supported for "
                 "direct transfer. Try using the ProjectTensorAux kernel to transfer buffers to "
                 "variables of any other type.");

    _buffer_to_var[buffer_name] = std::make_tuple(&var, std::vector<std::size_t>{}, is_nodal);
    getCPUBuffer(buffer_name);
  }
}

void
TensorProblem::addTensorComputeSolve(const std::string & compute_name,
                                     const std::string & name,
                                     InputParameters & parameters)
{
  addTensorCompute(compute_name, name, parameters, _computes);
}

void
TensorProblem::addTensorComputeInitialize(const std::string & compute_name,
                                          const std::string & name,
                                          InputParameters & parameters)
{
  addTensorCompute(compute_name, name, parameters, _ics);
}

void
TensorProblem::addTensorComputePostprocess(const std::string & compute_name,
                                           const std::string & name,
                                           InputParameters & parameters)
{
  addTensorCompute(compute_name, name, parameters, _pps);
}

void
TensorProblem::addTensorCompute(const std::string & compute_name,
                                const std::string & name,
                                InputParameters & parameters,
                                TensorComputeList & list)
{
  // Add a pointer to the TensorProblem and the Domain
  parameters.addPrivateParam<TensorProblem *>("_tensor_problem", this);
  parameters.addPrivateParam<const DomainAction *>("_domain", &_domain);

  // Create the object
  auto compute_object = _factory.create<TensorOperatorBase>(compute_name, name, parameters, 0);
  logAdd("TensorOperatorBase", name, compute_name, parameters);
  list.push_back(compute_object);
}

void
TensorProblem::addTensorTimeIntegrator(const std::string & time_integrator_name,
                                    const std::string & name,
                                    InputParameters & parameters)
{
  // Add a pointer to the TensorProblem and the Domain
  parameters.addPrivateParam<TensorProblem *>("_tensor_problem", this);
  parameters.addPrivateParam<const DomainAction *>("_domain", &_domain);

  // check that we have no other TI that advances the same buffer
  const auto & output_buffer = parameters.get<TensorOutputBufferName>("buffer");
  for (const auto & ti : _time_integrators)
    if (ti->parameters().get<TensorOutputBufferName>("buffer") == output_buffer)
      mooseError("Buffer '",
                 output_buffer,
                 "' is already advanced by time integrator '",
                 ti->name(),
                 "'. Cannot add '",
                 name,
                 "'.");

  // Create the object
  auto time_integrator_object =
      _factory.create<TensorTimeIntegrator>(time_integrator_name, name, parameters, 0);
  logAdd("TensorTimeIntegrator", name, time_integrator_name, parameters);
  _time_integrators.push_back(time_integrator_object);
}

void
TensorProblem::addTensorOutput(const std::string & output_name,
                            const std::string & name,
                            InputParameters & parameters)
{
  // Add a pointer to the TensorProblem and the Domain
  parameters.addPrivateParam<TensorProblem *>("_tensor_problem", this);
  parameters.addPrivateParam<const DomainAction *>("_domain", &_domain);

  // Create the object
  auto output_object = _factory.create<TensorOutput>(output_name, name, parameters, 0);
  logAdd("TensorInitialCondition", name, output_name, parameters);
  _outputs.push_back(output_object);
}

torch::Tensor &
TensorProblem::getBuffer(const std::string & buffer_name)
{
  auto it = _tensor_buffer.find(buffer_name);
  if (it == _tensor_buffer.end())
    mooseError("TensorBuffer '", buffer_name, "' does not exist in the system");
  return it->second;
}

const std::vector<torch::Tensor> &
TensorProblem::getBufferOld(const std::string & buffer_name, unsigned int max_states)
{
  auto it = _old_tensor_buffer.find(buffer_name);
  if (it == _old_tensor_buffer.end())
  {
    auto [newit, success] = _old_tensor_buffer.emplace(
        buffer_name, std::make_pair(max_states, std::vector<torch::Tensor>{}));
    if (success)
      it = newit;
    else
      mooseError("Failed to insert old buffer state");
  }
  else
    it->second.first = std::max(it->second.first, max_states);
  return it->second.second;
}

const torch::Tensor &
TensorProblem::getCPUBuffer(const std::string & buffer_name)
{
  // does the buffer we request to copy to the CPU actually exist?
  if (_tensor_buffer.find(buffer_name) == _tensor_buffer.end())
    mooseError("Buffer '", buffer_name, "' does not exist. Cannot request a CPU copy of it.");

  // add it to the list of buffers to be copied
  auto it = _tensor_cpu_buffer.find(buffer_name);
  if (it == _tensor_cpu_buffer.end())
  {
    auto [newit, success] = _tensor_cpu_buffer.try_emplace(buffer_name);
    if (success)
      it = newit;
    else
      mooseError("Failed to insert read-only CPU buffer");
  }
  return it->second;
}

void TensorProblem::setSolver(std::shared_ptr<TensorSolver> solver,
                              const MooseTensor::Key<CreateTensorSolverAction> &)
{
  if (_solver)
    mooseError("A solver has already been set up.");

  _solver = solver;

  // check that no legacy time integrators have been set
  if (!_time_integrators.empty())
    mooseError(
        "Do not supply any legacy TensorTimeIntegrators if a TensorSolver is given in the input.");
}

