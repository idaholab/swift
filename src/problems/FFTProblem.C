//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTProblem.h"
#include "FFTMesh.h"
#include "FFTCompute.h"
#include "FFTInitialCondition.h"
#include "FFTTimeIntegrator.h"
#include "SwiftUtils.h"
#include "DependencyResolverInterface.h"

registerMooseObject("SwiftApp", FFTProblem);

InputParameters
FFTProblem::validParams()
{
  InputParameters params = FEProblem::validParams();
  params.addClassDescription(
      "A normal Problem object that adds the ability to perform spectral solves.");
  params.set<bool>("skip_nl_system_check") = true;
  params.addParam<unsigned int>(
      "spectral_solve_substeps",
      1,
      "How many substeps to divide the spectral solve for each MOOSE timestep into.");
  return params;
}

FFTProblem::FFTProblem(const InputParameters & parameters)
  : FEProblem(parameters),
    _fft_mesh(dynamic_cast<FFTMesh *>(&_mesh)),
    _options(MooseFFT::floatTensorOptions()),
    _substeps(getParam<unsigned int>("spectral_solve_substeps"))
{
  if (!_fft_mesh)
    mooseError("FFTProblem must be used with an FFTMesh");
  _dim = _fft_mesh->getDim();

  // make sure AuxVariables are contiguous in teh solution vector
  getAuxiliarySystem().sys().identify_variable_groups(false);

  // this should only be run in serial, there is no COMM for the torch stuff yet
  if (comm().size() > 1)
    mooseError("FFT problems can only be run in serial at this time.");
}

void
FFTProblem::init()
{
  // get grid geometry
  for (const auto dim : make_range(3))
  {
    _max[dim] = _fft_mesh->getMaxInDimension(dim);
    _n[dim] = _fft_mesh->getElementsInDimension(dim);
    _grid_spacing[dim] = _max[dim] / _n[dim];
  }

  switch (_dim)
  {
    case 1:
      _shape_storage = {_n[0]};
      break;

    case 2:
      _shape_storage = {_n[0], _n[1]};
      break;

    case 3:
      _shape_storage = {_n[0], _n[1], _n[2]};
      break;

    default:
      mooseError("Unsupported mesh dimension");
  }
  _shape = _shape_storage;

  // initialize tensors (assuming all scalar for now, but in the future well have an FFTBufferBase
  // pointer as well)
  for (auto pair : _fft_buffer)
    pair.second = torch::zeros(_shape, _options);

  // build real space axes
  for (const auto dim : make_range(3))
  {
    if (dim < _dim)
      _axis[dim] = align(torch::linspace(c10::Scalar(_grid_spacing[dim] / 2.0),
                                         c10::Scalar(_max[dim] - _grid_spacing[dim] / 2.0),
                                         _n[dim],
                                         _options),
                         dim);
    else
      _axis[dim] = torch::tensor({0.0}, _options);
  }

  // build reciprocal space axes
  for (const auto dim : make_range(3))
  {
    if (dim < _dim)
    {
      const auto freq = (dim == _dim - 1)
                            ? torch::fft::rfftfreq(_n[dim], _grid_spacing[dim], _options)
                            : torch::fft::fftfreq(_n[dim], _grid_spacing[dim], _options);
      _reciprocal_axis[dim] = align(freq, dim);
    }
    else
      _reciprocal_axis[dim] = torch::tensor({0.0}, _options);
  }

  // dependency resolution of FFTICs
  DependencyResolverInterface::sort(_ics);

  // dependency resolution of FFTComputes
  DependencyResolverInterface::sort(_computes);

  // call base class init
  FEProblem::init();
}

void
FFTProblem::execute(const ExecFlagType & exec_type)
{
  if (exec_type == EXEC_INITIAL)
  {
    // run ICs
    for (auto & ic : _ics)
      ic->computeBuffer();
  }

  if (exec_type == EXEC_TIMESTEP_BEGIN)
  {
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

    mapBuffersToAux();
  }

  // if (exec_type == EXEC_TIMESTEP_END)
  //   // map buffers to AuxVariables
  //   mapBuffersToAux();

  FEProblem::execute(exec_type);
}

void
FFTProblem::mapBuffersToAux()
{
  auto * current_solution = &getAuxiliarySystem().solution();
  auto * solution_vector = dynamic_cast<PetscVector<Number> *>(current_solution);
  if (!solution_vector)
    mooseError(
        "Cannot map directly to the solution vector because NumericVector is not a PetscVector!");

  mooseInfoRepeated("solution local size is ", solution_vector->local_size());
  auto value = solution_vector->get_array();
  // do magic
  for (const auto i : make_range(solution_vector->local_size()))
    value[i] = 0.001 * i;
  solution_vector->restore_array();

  getAuxiliarySystem().sys().update();
}

void
FFTProblem::advanceState()
{
  FEProblem::advanceState();

  if (timeStep() <= 1)
    return;

  // move buffers in time
  for (auto & [name, max_states] : _old_fft_buffer)
  {
    auto & [max, states] = max_states;
    if (states.size() < max)
      states.push_back(torch::tensor({}, _options));
    for (std::size_t i = states.size() - 1; i > 0; --i)
      states[i] = states[i - 1];
    states[0] = _fft_buffer[name];
  }
}

void
FFTProblem::addFFTBuffer(const std::string & buffer_name, InputParameters & parameters)
{
  if (_fft_buffer.find(buffer_name) != _fft_buffer.end())
    mooseError("FFTBuffer '", buffer_name, "' already exists in the system");
  _fft_buffer.try_emplace(buffer_name);
}

void
FFTProblem::addFFTCompute(const std::string & compute_name,
                          const std::string & name,
                          InputParameters & parameters)
{
  // Add a pointer to the FFTProblem class
  parameters.addPrivateParam<FFTProblem *>("_fft_problem", this);

  // Create the object
  std::shared_ptr<FFTCompute> compute_object =
      _factory.create<FFTCompute>(compute_name, name, parameters, 0);
  logAdd("FFTCompute", name, compute_name);
  _computes.push_back(compute_object);
}

void
FFTProblem::addFFTIC(const std::string & ic_name,
                     const std::string & name,
                     InputParameters & parameters)
{
  // Add a pointer to the FFTProblem class
  parameters.addPrivateParam<FFTProblem *>("_fft_problem", this);

  // Create the object
  std::shared_ptr<FFTInitialCondition> ic_object =
      _factory.create<FFTInitialCondition>(ic_name, name, parameters, 0);
  logAdd("FFTInitialCondition", name, ic_name);
  _ics.push_back(ic_object);
}

void
FFTProblem::addFFTTimeIntegrator(const std::string & time_integrator_name,
                                 const std::string & name,
                                 InputParameters & parameters)
{
  // Add a pointer to the FFTProblem class
  parameters.addPrivateParam<FFTProblem *>("_fft_problem", this);

  // check that we have no other TI that advances the same buffer
  const auto & output_buffer = parameters.get<FFTOutputBufferName>("buffer");
  for (const auto ti : _time_integrators)
    if (ti->parameters().get<FFTOutputBufferName>("buffer") == output_buffer)
      mooseError("Buffer '",
                 output_buffer,
                 "' is already advanced by time integrator '",
                 ti->name(),
                 "'. Cannot add '",
                 name,
                 "'.");

  // Create the object
  std::shared_ptr<FFTTimeIntegrator> time_integrator_object =
      _factory.create<FFTTimeIntegrator>(time_integrator_name, name, parameters, 0);
  logAdd("FFTTimeIntegrator", name, time_integrator_name);
  _time_integrators.push_back(time_integrator_object);
}

torch::Tensor &
FFTProblem::getBuffer(const std::string & buffer_name)
{
  auto it = _fft_buffer.find(buffer_name);
  if (it == _fft_buffer.end())
    mooseError("FFTBuffer '", buffer_name, "' does not exist in the system");
  return it->second;
}

const std::vector<torch::Tensor> &
FFTProblem::getBufferOld(const std::string & buffer_name, unsigned int max_states)
{
  auto it = _old_fft_buffer.find(buffer_name);
  if (it == _old_fft_buffer.end())
  {
    auto [newit, success] = _old_fft_buffer.emplace(
        buffer_name, std::make_pair(max_states, std::vector<torch::Tensor>{}));
    if (success)
      it = newit;
    else
      mooseError("Failed to insert old buffer state");
  }
  return it->second.second;
}

const torch::Tensor &
FFTProblem::getAxis(std::size_t component) const
{
  if (component < 3)
    return _axis[component];
  mooseError("Invalid component");
}

const torch::Tensor &
FFTProblem::getReciprocalAxis(std::size_t component) const
{
  if (component < 3)
    return _reciprocal_axis[component];
  mooseError("Invalid component");
}

torch::Tensor
FFTProblem::fft(torch::Tensor t) const
{
  switch (_dim)
  {
    case 1:
      return torch::fft::rfft(t);
    case 2:
      return torch::fft::rfft2(t);
    case 3:
      return torch::fft::rfftn(t, c10::nullopt, {0, 1, 2});
    default:
      mooseError("Unsupported mesh dimension");
  }
}

torch::Tensor
FFTProblem::ifft(torch::Tensor t) const
{
  switch (_dim)
  {
    case 1:
      return torch::fft::irfft(t);
    case 2:
      return torch::fft::irfft2(t);
    case 3:
      return torch::fft::irfftn(t, c10::nullopt, {0, 1, 2});
    default:
      mooseError("Unsupported mesh dimension");
  }
}

torch::Tensor
FFTProblem::align(torch::Tensor t, unsigned int dim) const
{
  if (dim >= _dim)
    mooseError("Unsupported alignment dimension requested dimension");

  switch (_dim)
  {
    case 1:
      return t;

    case 2:
      if (dim == 0)
        return torch::unsqueeze(t, 1);
      else
        return torch::unsqueeze(t, 0);

    case 3:
      if (dim == 0)
        return t.unsqueeze(1).unsqueeze(2);
      else if (dim == 1)
        return t.unsqueeze(0).unsqueeze(2);
      else
        return t.unsqueeze(0).unsqueeze(0);

    default:
      mooseError("Unsupported mesh dimension");
  }
}
