/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "FEProblem.h"
#include "DomainInterface.h"
#include "SwiftTypes.h"
#include "SwiftUtils.h"
#include "TensorBuffer.h"

#include "AuxiliarySystem.h"
#include "libmesh/petsc_vector.h"

#include <memory>
#include <torch/torch.h>

class UniformTensorMesh;
class TensorOperatorBase;
class TensorTimeIntegrator;
class TensorOutput;
class TensorSolver;
class CreateTensorSolverAction;

/**
 * Problem for solving eigenvalue problems
 */
class TensorProblem : public FEProblem, public DomainInterface
{
public:
  static InputParameters validParams();

  TensorProblem(const InputParameters & parameters);
  ~TensorProblem() override;

  // setup stuff
  void init() override;

  // run compute objects
  void execute(const ExecFlagType & exec_type) override;

  // move tensors in time
  void advanceState() override;

  // recompute quantities on grid size change
  virtual void gridChanged();

  virtual void addTensorBuffer(const std::string & buffer_type,
                               const std::string & buffer_name,
                               InputParameters & parameters);

  virtual void addTensorComputeInitialize(const std::string & compute_name,
                                          const std::string & name,
                                          InputParameters & parameters);
  virtual void addTensorComputeSolve(const std::string & compute_name,
                                     const std::string & name,
                                     InputParameters & parameters);
  virtual void addTensorComputePostprocess(const std::string & compute_name,
                                           const std::string & name,
                                           InputParameters & parameters);
  virtual void addTensorComputeOnDemand(const std::string & compute_name,
                                        const std::string & name,
                                        InputParameters & parameters);

  virtual void addTensorTimeIntegrator(const std::string & time_integrator_name,
                                       const std::string & name,
                                       InputParameters & parameters);
  virtual void addTensorOutput(const std::string & output_name,
                               const std::string & name,
                               InputParameters & parameters);

  /// returns teh current state of the tensor
  template <typename T = torch::Tensor>
  T & getBuffer(const std::string & buffer_name);

  /// return the old states of the tensor
  template <typename T = torch::Tensor>
  const std::vector<T> & getBufferOld(const std::string & buffer_name, unsigned int max_states);

  /// returns a reference to a copy of buffer_name that is guaranteed to be contiguous and located on the CPU device
  template <typename T = torch::Tensor>
  const T & getCPUBuffer(const std::string & buffer_name);

  TensorOperatorBase & getOnDemandCompute(const std::string & name);

  virtual Real & subDt() { return _sub_dt; }
  virtual Real & subTime() { return _sub_time; }
  virtual Real & outputTime() { return _output_time; }

  /// align a 1d tensor in a specific dimension
  torch::Tensor align(torch::Tensor t, unsigned int dim) const;

  /// get the domain shape (to build tensors from scratch) TODO: make sure this is local
  const torch::IntArrayRef & getShape() { return _shape; }

  typedef std::vector<std::shared_ptr<TensorOperatorBase>> TensorComputeList;
  const TensorComputeList & getComputes() const { return _computes; }

  typedef std::vector<std::shared_ptr<TensorOutput>> TensorOutputList;
  const TensorOutputList & getOutputs() const { return _outputs; }

  /// The CreateTensorSolverAction calls this to set the active solver
  void setSolver(std::shared_ptr<TensorSolver> solver,
                 const MooseTensor::Key<CreateTensorSolverAction> &);

  /// get a reference to the current solver
  template <typename T>
  T & getSolver() const;

protected:
  void updateDOFMap();

  template <typename FLOAT_TYPE>
  void mapBuffersToAux();

  template <typename FLOAT_TYPE>
  void mapAuxToBuffers();

  virtual void addTensorCompute(const std::string & compute_name,
                                const std::string & name,
                                InputParameters & parameters,
                                TensorComputeList & list);

  /// execute initial conditionobjects
  void executeTensorInitialConditions();

  /// perform output tasks
  void executeTensorOutputs(const ExecFlagType & exec_type);

  /// helper to get the TensorBuffer wrapper object that holds the actual tensor data
  template <typename T = torch::Tensor>
  TensorBuffer<T> & getBufferHelper(const std::string & buffer_name);

  /// tensor options
  const torch::TensorOptions _options;

  /// show debug ouput
  const bool _debug;

  /// solver substeps
  const unsigned int _substeps;

  /// substepping timestep
  Real _sub_dt;
  Real _sub_time;

  /// simulation time for the currently running output thread
  Real _output_time;

  /// list of TensorBuffers (i.e. tensors)
  std::map<std::string, std::shared_ptr<TensorBufferBase>> _tensor_buffer;

  /// set of tensors that need to be copied to the CPU
  std::set<std::string> _cpu_tensor_buffers;

  /// old timesteps
  std::vector<Real> _old_dt;

  const unsigned int & _dim;

  /// grid spacing
  const RealVectorValue & _grid_spacing;

  /// global grid size
  const std::array<int64_t, 3> & _n;

  /// domain shape
  const torch::IntArrayRef & _shape;

  /// solve objects
  TensorComputeList _computes;

  /// initialization objects
  TensorComputeList _ics;

  /// postprocessing objects
  TensorComputeList _pps;

  /// on demand objects that are explicitly triggered by other objects
  TensorComputeList _on_demand;

  ///  time integrator objects
  std::vector<std::shared_ptr<TensorTimeIntegrator>> _time_integrators;

  std::vector<std::shared_ptr<TensorOutput>> _outputs;

  /// map from buffer name to variable name
  std::map<std::string, AuxVariableName> _buffer_to_var_name;

  /// buffers to solution vector indices
  std::map<std::string, std::tuple<const MooseVariableFieldBase *, std::vector<std::size_t>, bool>>
      _buffer_to_var;
  std::map<std::string, std::tuple<const MooseVariableFieldBase *, std::vector<std::size_t>, bool>>
      _var_to_buffer;

  /// The [TensorSolver]
  std::shared_ptr<TensorSolver> _solver;
};

#include "TensorSolver.h"

template <typename T>
T &
TensorProblem::getSolver() const
{
  if (_solver)
  {
    const auto specialized_solver = dynamic_cast<T *>(_solver.get());
    if (specialized_solver)
      return *specialized_solver;
    mooseError(
        "No TensorSolver supporting the requested type '", typeid(T).name(), "' has been set up.");
  }
  mooseError("No TensorSolver has been set up.");
}

template <typename T>
TensorBuffer<T> &
TensorProblem::getBufferHelper(const std::string & buffer_name)
{
  auto it = _tensor_buffer.find(buffer_name);
  if (it == _tensor_buffer.end())
    mooseError("TensorBuffer '", buffer_name, "' does not exist in the system.");
  auto tensor_buffer = dynamic_cast<TensorBuffer<T> *>(it->second.get());
  if (!tensor_buffer)
    mooseError("TensorBuffer '",
               buffer_name,
               "' of the requested type '",
               it->second->type(),
               "' does not exist in the system.");
  return *tensor_buffer;
}

template <typename T>
T &
TensorProblem::getBuffer(const std::string & buffer_name)
{
  return getBufferHelper<T>(buffer_name)._u;
}

template <typename T>
const std::vector<T> &
TensorProblem::getBufferOld(const std::string & buffer_name, unsigned int max_states)
{
  auto & tensor_buffer = getBufferHelper<T>(buffer_name);

  if (tensor_buffer._max_states < max_states)
    tensor_buffer._max_states = max_states;

  return tensor_buffer._u_old;
}

template <typename T>
const T &
TensorProblem::getCPUBuffer(const std::string & buffer_name)
{
  _cpu_tensor_buffers.insert(buffer_name);
  return getBufferHelper<T>(buffer_name)._u_cpu;
}
