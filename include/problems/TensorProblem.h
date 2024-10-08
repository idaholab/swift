//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#ifdef NEML2_ENABLED

#include "FEProblem.h"
#include "DomainInterface.h"
#include "SwiftTypes.h"

#include "AuxiliarySystem.h"
#include "libmesh/petsc_vector.h"

#include "torch/torch.h"

class UniformTensorMesh;
class TensorOperatorBase;
class TensorTimeIntegrator;
class TensorOutput;

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

  virtual void addTensorBuffer(const std::string & buffer_name, InputParameters & parameters);

  virtual void addTensorComputeInitialize(const std::string & compute_name,
                                          const std::string & name,
                                          InputParameters & parameters);
  virtual void addTensorComputeSolve(const std::string & compute_name,
                                     const std::string & name,
                                     InputParameters & parameters);
  virtual void addTensorComputePostprocess(const std::string & compute_name,
                                           const std::string & name,
                                           InputParameters & parameters);

  virtual void addTensorTimeIntegrator(const std::string & time_integrator_name,
                                    const std::string & name,
                                    InputParameters & parameters);
  virtual void addTensorOutput(const std::string & output_name,
                            const std::string & name,
                            InputParameters & parameters);

  torch::Tensor & getBuffer(const std::string & buffer_name);
  const std::vector<torch::Tensor> & getBufferOld(const std::string & buffer_name,
                                                  unsigned int max_states);

  /// returns a reference to a copy of buffer_name that is guaranteed to be contiguous and located on the CPU device
  const torch::Tensor & getCPUBuffer(const std::string & buffer_name);

  const unsigned int & getDim() const { return _domain.getDim(); }
  const Real & getSubDt() const { return _sub_dt; }

  const std::array<int64_t, 3> & getGridSize() const { return _domain.getGridSize(); }
  const std::array<Real, 3> & getGridSpacing() const { return _domain.getGridSpacing(); }
  const torch::Tensor & getAxis(std::size_t component) const { return _domain.getAxis(component); }
  const torch::Tensor & getReciprocalAxis(std::size_t component) const
  {
    return _domain.getReciprocalAxis(component);
  }

  const torch::Tensor & getKSquare() const { return _domain.getKSquare(); }

  torch::Tensor fft(torch::Tensor t) const { return _domain.fft(t); }
  torch::Tensor ifft(torch::Tensor t) const { return _domain.ifft(t); }

  /// align a 1d tensor in a specific dimension
  torch::Tensor align(torch::Tensor t, unsigned int dim) const;

  /// get the domain shape (to build tensors from scratch)
  const torch::IntArrayRef & getShape() { return _shape; }

protected:
  void updateDOFMap();
  void mapBuffersToAux();

  typedef std::vector<std::shared_ptr<TensorOperatorBase>> TensorComputeList;

  virtual void addTensorCompute(const std::string & compute_name,
                                const std::string & name,
                                InputParameters & parameters,
                                TensorComputeList & list);

  /// Domain
  const DomainAction & _domain;

  /// tensor options
  const torch::TensorOptions _options;

  /// show debug ouput
  const bool _debug;

  /// solver substeps
  const unsigned int _substeps;

  /// substepping timestep
  Real _sub_dt;

  /// list of TensorBuffers (i.e. tensors)
  std::map<std::string, torch::Tensor> _tensor_buffer;

  /// list of read-only CPU TensorBuffers (for MOOSE objects and outputs)
  std::map<std::string, torch::Tensor> _tensor_cpu_buffer;

  /// old buffers (stores max number of states, requested, and states)
  std::map<std::string, std::pair<unsigned int, std::vector<torch::Tensor>>> _old_tensor_buffer;

  const unsigned int & _dim;

  /// grid spacing
  const std::array<Real, 3> & _grid_spacing;

  /// global grid size
  const std::array<int64_t, 3> & _n;

  /// domain shape
  torch::IntArrayRef _shape;

  /// solve objects
  TensorComputeList _computes;

  /// initialization objects
  TensorComputeList _ics;

  /// postprocessing objects
  TensorComputeList _pps;

  ///  time integrator objects
  std::vector<std::shared_ptr<TensorTimeIntegrator>> _time_integrators;

  std::vector<std::shared_ptr<TensorOutput>> _outputs;

  /// map from buffer name to variable name
  std::map<std::string, AuxVariableName> _buffer_to_var_name;

  /// buffers to solution vector indices
  std::map<std::string, std::tuple<const MooseVariableFieldBase *, std::vector<std::size_t>, bool>>
      _buffer_to_var;
};

#endif
