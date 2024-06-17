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
#include "SwiftTypes.h"
#include "torch/torch.h"

class FFTMesh;
class FFTCompute;
class FFTInitialCondition;
class FFTTimeIntegrator;

/**
 * Problem for solving eigenvalue problems
 */
class FFTProblem : public FEProblem
{
public:
  static InputParameters validParams();

  FFTProblem(const InputParameters & parameters);

  // setup stuff
  void init() override;

  // run compute objects
  void execute(const ExecFlagType & exec_type) override;

  // move tensors in time
  void advanceState() override;

  virtual void addFFTBuffer(const std::string & buffer_name, InputParameters & parameters);
  virtual void addFFTCompute(const std::string & compute_name,
                             const std::string & name,
                             InputParameters & parameters);
  virtual void addFFTIC(const std::string & compute_name,
                        const std::string & name,
                        InputParameters & parameters);
  virtual void addFFTTimeIntegrator(const std::string & time_integrator_name,
                                    const std::string & name,
                                    InputParameters & parameters);

  torch::Tensor & getBuffer(const std::string & buffer_name);
  const std::vector<torch::Tensor> & getBufferOld(const std::string & buffer_name,
                                                  unsigned int max_states);

  unsigned int & getDim() { return _dim; }
  const std::array<unsigned int, 3> & getGridSize() const { return _n; }
  const std::array<Real, 3> & getGridSpacing() const { return _grid_spacing; }
  const torch::Tensor & getAxis(std::size_t component) const;

  torch::Tensor fft(torch::Tensor t) const;
  torch::Tensor ifft(torch::Tensor t) const;

protected:
  /// FFT Mesh object
  FFTMesh * _fft_mesh;

  /// tensor options
  const torch::TensorOptions _options;

  /// list of FFTBuffers (i.e. tensors)
  std::map<std::string, torch::Tensor> _fft_buffer;

  /// old buffers (stores max number of states, requested, and states)
  std::map<std::string, std::pair<unsigned int, std::vector<torch::Tensor>>> _old_fft_buffer;

  unsigned int _dim;

  /// Number of elements in x, y, z direction
  std::array<unsigned int, 3> _n;

  /// The max values for x,y,z component
  std::array<Real, 3> _max;

  /// grid spacing
  std::array<Real, 3> _grid_spacing;

  /// domain shape
  std::vector<long int> _shape_storage;
  torch::IntArrayRef _shape;

  /// real space axes
  std::array<torch::Tensor, 3> _axis;

  /// reciprocal space axes
  std::array<torch::Tensor, 3> _reciprocal_axis;

  /// compute objects
  std::vector<std::shared_ptr<FFTCompute>> _computes;

  /// ic objects
  std::vector<std::shared_ptr<FFTInitialCondition>> _ics;

  ///  time integrator objects
  std::vector<std::shared_ptr<FFTTimeIntegrator>> _time_integrators;
};

#endif
