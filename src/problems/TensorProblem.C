/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "Moose.h"
#include "MooseError.h"
#include "TensorProblem.h"
#include "TensorSolver.h"
#include "UniformTensorMesh.h"

#include "TensorOperatorBase.h"
#include "TensorTimeIntegrator.h"
#include "TensorOutput.h"
#include "DomainAction.h"

#include "SwiftUtils.h"
#include "DependencyResolverInterface.h"
#include <memory>

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
  params.addParam<std::vector<std::string>>("scalar_constant_names", "Scalar constant names");
  params.addParam<std::vector<Real>>("scalar_constant_values", "Scalar constant values");
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
    _solver(nullptr),
    _can_fetch_constants(true)
{
  // get constants (for scalar constants we provide a shortcut in the problem block)
  for (const auto & [name, value] :
       getParam<std::string, Real>("scalar_constant_names", "scalar_constant_values"))
    declareConstant<Real>(name, value);

  // make sure AuxVariables are contiguous in the solution vector
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
  unsigned int n_threads = libMesh::n_threads();
  if (n_threads != 1)
  {
    mooseInfo("Setting libTorch to use ", n_threads, " threads on the CPU.");
    torch::set_num_threads(n_threads);
  }

  // initialize tensors
  for (auto pair : _tensor_buffer)
    pair.second->init();

  // compute grid dependent quantities
  gridChanged();

  // init computes (must happen before dependency update)
  for (auto & cmp : _computes)
    cmp->init();

  // update dependencies
  if (_solver)
    _solver->updateDependencies();
  else
  {
    // dependency resolution of TensorComputes
    DependencyResolverInterface::sort(_computes);
  }

  // dependency resolution of TensorICs
  DependencyResolverInterface::sort(_ics);

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

  // check computes (after dependency update)
  for (auto & cmp : _computes)
    cmp->check();

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
    // check for constants
    if (_fetched_constants.size() == 1)
      mooseError(
          "Constant ", Moose::stringify(_fetched_constants), " was requested but never declared.");
    if (_fetched_constants.size() > 1)
      mooseError("Constants ",
                 Moose::stringify(_fetched_constants),
                 " were requested but never declared.");
    _can_fetch_constants = false;

    // update time
    _sub_time = FEProblem::time();

    executeTensorInitialConditions();

    executeTensorOutputs(EXEC_INITIAL);
  }

  if (exec_type == EXEC_TIMESTEP_BEGIN)
  {
    // update time
    _sub_time = FEProblem::timeOld();

    // run solver
    if (_solver)
      _solver->computeBuffer();
    else
      for (auto & cmp : _computes)
        cmp->computeBuffer();
  }

  if (exec_type == EXEC_TIMESTEP_END)
  {
    // run outputs
    executeTensorOutputs(EXEC_TIMESTEP_END);
  }

  FEProblem::execute(exec_type);
}

void
TensorProblem::executeTensorInitialConditions()
{
  // run ICs
  for (auto & ic : _ics)
    ic->computeBuffer();

  // compile ist of compute output tensors
  std::set<std::string> _is_output;
  for (auto & cmp : _computes)
    _is_output.insert(cmp->getSuppliedItems().begin(), cmp->getSuppliedItems().end());

  // // check for uninitialized tensors
  // for (auto & [name, t] : _tensor_buffer)
  //   if (!t.defined() && _is_output.count(name) == 0)
  //     mooseWarning(name, " is not initialized and not an output of any [Solve] compute.");
}

/// perform output tasks
void
TensorProblem::executeTensorOutputs(const ExecFlagType & exec_flag)
{
  // run postprocessing before output
  for (auto & pp : _pps)
    pp->computeBuffer();

  // wait for prior asynchronous activity on CPU buffers to complete
  // (this is a synchronization barrier for the threaded CPU activity)
  for (auto & output : _outputs)
    output->waitForCompletion();

  // update output time
  _output_time = _time;

  // prepare CPU buffers (this is a synchronization barrier for the GPU)
  for (const auto & pair : _tensor_buffer)
    pair.second->makeCPUCopy();

  // run direct buffer outputs (asynchronous in threads)
  for (auto & output : _outputs)
    if (output->shouldRun(exec_flag))
      output->startOutput();

  if (_options.dtype() == torch::kFloat64)
    mapBuffersToAux<double>();
  else if (_options.dtype() == torch::kFloat32)
    mapBuffersToAux<float>();
  else
    mooseError("torch::Dtype unsupported by mapBuffersToAux.");
}

void
TensorProblem::updateDOFMap()
{
  TIME_SECTION("update", 3, "Updating Tensor DOF Map", true);
  const auto & min_global = _domain.getDomainMin();

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

    auto compute_iteration_index = [this](Point p, long int n0, long int n1)
    {
      return static_cast<long int>(p(0) / _grid_spacing(0)) +
             (_dim > 1 ? static_cast<long int>(p(1) / _grid_spacing(1)) * n0 : 0) +
             (_dim > 2 ? static_cast<long int>(p(2) / _grid_spacing(2)) * n0 * n1 : 0);
    };

    if (is_nodal)
    {
      long int n0 = _n[0] + 1;
      long int n1 = _n[1] + 1;
      long int n2 = _n[2] + 1;
      dofs.resize(n0 * (_dim > 1 ? n1 : 1) * (_dim > 2 ? n2 : 1));

      // loop over nodes
      const static Point shift = _grid_spacing / 2.0 - min_global;
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
      const static Point shift = -min_global;
      for (const auto & elem : _mesh.getMesh().element_ptr_range())
      {
        const auto dof_index = elem->dof_number(sys_num, var_num, 0);
        const auto iteration_index =
            compute_iteration_index(elem->vertex_average() + shift, n0, n1);
        dofs[iteration_index] = dof_index;
      }
    }
  }
}

template <typename FLOAT_TYPE>
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

    // TODO: better design that works for NEML2 tensors as well
    const auto buffer = getRawCPUBuffer(buffer_name);
    if (buffer.sizes().size() != _dim)
      mooseError("Buffer '",
                 buffer_name,
                 "' is not a scalar tensor field and is not yet supported for AuxVariable mapping");
    std::size_t idx = 0;
    switch (_dim)
    {
      {
        case 1:
          const auto b = buffer.template accessor<FLOAT_TYPE, 1>();
          for (const auto i : make_range(n0))
            value[dofs[idx++]] = b[i % _n[0]];
          break;
      }
      case 2:
      {
        const auto b = buffer.template accessor<FLOAT_TYPE, 2>();
        for (const auto j : make_range(n1))
          for (const auto i : make_range(n0))
            value[dofs[idx++]] = b[i % _n[0]][j % _n[1]];
        break;
      }
      case 3:
      {
        const auto b = buffer.template accessor<FLOAT_TYPE, 3>();
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

template <typename FLOAT_TYPE>
void
TensorProblem::mapAuxToBuffers()
{
  // nothing to map?
  if (_var_to_buffer.empty())
    return;

  TIME_SECTION("update", 3, "Mapping Variables to Tensor buffers", true);

  const auto * current_solution = &getAuxiliarySystem().solution();
  const auto * solution_vector = dynamic_cast<const PetscVector<Number> *>(current_solution);
  if (!solution_vector)
    mooseError(
        "Cannot map directly to the solution vector because NumericVector is not a PetscVector!");

  const auto value = solution_vector->get_array_read();

  // const monomial variables
  for (const auto & [buffer_name, tuple] : _var_to_buffer)
  {
    const auto & [var, dofs, is_nodal] = tuple;
    libmesh_ignore(var);
    const auto buffer = getBufferBase(buffer_name).getRawCPUTensor();
    std::size_t idx = 0;
    switch (_dim)
    {
      {
        case 1:
          auto b = buffer.template accessor<FLOAT_TYPE, 1>();
          for (const auto i : make_range(_n[0]))
            b[i % _n[0]] = value[dofs[idx++]];
          break;
      }
      case 2:
      {
        auto b = buffer.template accessor<FLOAT_TYPE, 2>();
        for (const auto j : make_range(_n[1]))
        {
          for (const auto i : make_range(_n[0]))
            b[i % _n[0]][j % _n[1]] = value[dofs[idx++]];
          if (is_nodal)
            idx++;
        }
        break;
      }
      case 3:
      {
        auto b = buffer.template accessor<FLOAT_TYPE, 3>();
        for (const auto k : make_range(_n[2]))
        {
          for (const auto j : make_range(_n[1]))
          {
            for (const auto i : make_range(_n[0]))
              b[i % _n[0]][j % _n[1]][k % _n[2]] = value[dofs[idx++]];
            if (is_nodal)
              idx++;
          }
          if (is_nodal)
            idx += _n[0] + 1;
        }
        break;
      }
      default:
        mooseError("Unsupported dimension");
    }
  }
}

void
TensorProblem::advanceState()
{
  FEProblem::advanceState();

  if (timeStep() <= 1)
    return;

  // move buffers in time
  std::size_t total_max = 0;
  for (auto & pair : _tensor_buffer)
    total_max = std::max(total_max, pair.second->advanceState());

  // move dt in time (UGH, we need the _substep_dt!!!!)
  if (_old_dt.size() < total_max)
    _old_dt.push_back(0.0);
  if (!_old_dt.empty())
  {
    for (std::size_t i = _old_dt.size() - 1; i > 0; --i)
      _old_dt[i] = _old_dt[i - 1];
    _old_dt[0] = _dt;
  }
}

void
TensorProblem::gridChanged()
{
  // _domain.gridChanged();
}

void
TensorProblem::addTensorBuffer(const std::string & buffer_type,
                               const std::string & buffer_name,
                               InputParameters & parameters)
{
  // add buffer
  if (_tensor_buffer.find(buffer_name) != _tensor_buffer.end())
    mooseError("TensorBuffer '", buffer_name, "' already exists in the system");

  // Add a pointer to the TensorProblem and the Domain
  // parameters.addPrivateParam<TensorProblem *>("_tensor_problem", this);
  // parameters.addPrivateParam<const DomainAction *>("_domain", &_domain);

  // Create the object
  auto tensor_buffer = _factory.create<TensorBufferBase>(buffer_type, buffer_name, parameters, 0);
  logAdd("TensorBufferBase", buffer_name, buffer_type, parameters);

  _tensor_buffer.try_emplace(buffer_name, tensor_buffer);

  // store variable mapping
  const auto & var_names = parameters.get<std::vector<AuxVariableName>>("map_to_aux_variable");
  if (!var_names.empty())
  {
    const auto & aux = getAuxiliarySystem();
    const auto var_name = var_names[0];
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

    // call this to mark the CPU copy as requested
    getRawCPUBuffer(buffer_name);
  }
}

void
TensorProblem::addTensorComputeSolve(const std::string & compute_type,
                                     const std::string & compute_name,
                                     InputParameters & parameters)
{
  addTensorCompute(compute_type, compute_name, parameters, _computes);
}

void
TensorProblem::addTensorComputeInitialize(const std::string & compute_type,
                                          const std::string & compute_name,
                                          InputParameters & parameters)
{
  addTensorCompute(compute_type, compute_name, parameters, _ics);
}

void
TensorProblem::addTensorComputePostprocess(const std::string & compute_name,
                                           const std::string & name,
                                           InputParameters & parameters)
{
  addTensorCompute(compute_name, name, parameters, _pps);
}

void
TensorProblem::addTensorCompute(const std::string & compute_type,
                                const std::string & compute_name,
                                InputParameters & parameters,
                                TensorComputeList & list)
{
  // Add a pointer to the TensorProblem and the Domain
  parameters.addPrivateParam<TensorProblem *>("_tensor_problem", this);
  parameters.addPrivateParam<const DomainAction *>("_domain", &_domain);

  // Create the object
  auto compute_object =
      _factory.create<TensorOperatorBase>(compute_type, compute_name, parameters, 0);
  logAdd("TensorOperatorBase", compute_name, compute_type, parameters);
  list.push_back(compute_object);
}

void
TensorProblem::addTensorOutput(const std::string & output_type,
                               const std::string & output_name,
                               InputParameters & parameters)
{
  // Add a pointer to the TensorProblem and the Domain
  parameters.addPrivateParam<TensorProblem *>("_tensor_problem", this);
  parameters.addPrivateParam<const DomainAction *>("_domain", &_domain);

  // Create the object
  auto output_object = _factory.create<TensorOutput>(output_type, output_name, parameters, 0);
  logAdd("TensorInitialCondition", output_name, output_type, parameters);
  _outputs.push_back(output_object);
}

void
TensorProblem::setSolver(std::shared_ptr<TensorSolver> solver,
                         const MooseTensor::Key<CreateTensorSolverAction> &)
{
  if (_solver)
    mooseError("A solver has already been set up.");

  _solver = solver;
}

TensorBufferBase &
TensorProblem::getBufferBase(const std::string & buffer_name)
{
  auto it = _tensor_buffer.find(buffer_name);
  if (it == _tensor_buffer.end())
    mooseError("TensorBuffer '", buffer_name, " does not exist in the system.");
  return *it->second.get();
}

const torch::Tensor &
TensorProblem::getRawBuffer(const std::string & buffer_name)
{
  return getBufferBase(buffer_name).getRawTensor();
}

const torch::Tensor &
TensorProblem::getRawCPUBuffer(const std::string & buffer_name)
{
  return getBufferBase(buffer_name).getRawCPUTensor();
}
