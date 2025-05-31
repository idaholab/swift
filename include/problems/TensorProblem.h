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

// list tensor buffer includes here
#include "PlainTensorBuffer.h"
#include "NEML2TensorBuffer.h"

#include "AuxiliarySystem.h"
#include "libmesh/petsc_vector.h"
#include "libmesh/print_trace.h"

#include <memory>
#include <torch/torch.h>

class UniformTensorMesh;
class TensorOperatorBase;
template <typename T = torch::Tensor>
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

  template <typename T>
  std::shared_ptr<TensorBuffer<T>> addTensorBuffer(const std::string & buffer_name);

  virtual void addTensorComputeInitialize(const std::string & compute_name,
                                          const std::string & name,
                                          InputParameters & parameters);
  virtual void addTensorComputeSolve(const std::string & compute_name,
                                     const std::string & name,
                                     InputParameters & parameters);
  virtual void addTensorComputePostprocess(const std::string & compute_name,
                                           const std::string & name,
                                           InputParameters & parameters);

  virtual void addTensorOutput(const std::string & output_name,
                               const std::string & name,
                               InputParameters & parameters);

  /// returns the current state of the tensor
  template <typename T = torch::Tensor>
  T & getBuffer(const std::string & buffer_name);

  /// requests a tensor regardless of type
  TensorBufferBase & getBufferBase(const std::string & buffer_name);

  /// return the old states of the tensor
  template <typename T = torch::Tensor>
  const std::vector<T> & getBufferOld(const std::string & buffer_name, unsigned int max_states);

  /// returns a reference to a copy of buffer_name that is guaranteed to be contiguous and located on the CPU device
  const torch::Tensor & getRawCPUBuffer(const std::string & buffer_name);

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

  /// tensor outputs
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
    return *addTensorBuffer<T>(buffer_name);
  else
  {
    auto tensor_buffer = dynamic_cast<TensorBuffer<T> *>(it->second.get());
    if (!tensor_buffer)
      mooseError("TensorBuffer '",
                 buffer_name,
                 "' of the requested type '",
                 libMesh::demangle(typeid(T).name()),
                 "' was previously declared as '",
                 it->second->type(),
                 "'.");
    return *tensor_buffer;
  }
}

template <typename T>
T &
TensorProblem::getBuffer(const std::string & buffer_name)
{
  return getBufferHelper<T>(buffer_name).getTensor();
}

template <typename T>
const std::vector<T> &
TensorProblem::getBufferOld(const std::string & buffer_name, unsigned int max_states)
{
  return getBufferHelper<T>(buffer_name).getOldTensor(max_states);
}

template <typename T>
std::shared_ptr<TensorBuffer<T>>
TensorProblem::addTensorBuffer(const std::string & buffer_name)
{
  if (_debug)
    mooseInfoRepeated("Automatically adding tensor '",
                      buffer_name,
                      "' of type '",
                      libMesh::demangle(typeid(T).name()),
                      "'");
  auto params = TensorBuffer<T>::validParams();
  params.template set<std::string>("_object_name") = buffer_name;
  params.template set<FEProblem *>("_fe_problem") = this;
  params.template set<FEProblemBase *>("_fe_problem_base") = this;
  params.template set<THREAD_ID>("_tid") = 0;
  params.template set<std::string>("_type") = "TensorBufferBase";
  params.template set<MooseApp *>("_moose_app") = &getMooseApp();
  params.finalize(buffer_name);

  // params.addPrivateParam<TensorProblem *>("_tensor_problem", this);
  // params.addPrivateParam<const DomainAction *>("_domain", &_domain);

  auto tensor_buffer = std::make_shared<typename TensorBufferSpecialization<T>::type>(params);

  _tensor_buffer.try_emplace(buffer_name, tensor_buffer);
  return tensor_buffer;
}
