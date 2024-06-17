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
#include "FFTCompute.h"
#include "FFTInitialCondition.h"
#include "torch/torch.h"

/**
 * Problem for solving eigenvalue problems
 */
class FFTProblem : public FEProblem
{
public:
  static InputParameters validParams();

  FFTProblem(const InputParameters & parameters);

  void initialSetup() override;

  virtual void addFFTBuffer(const std::string & buffer_name, InputParameters & parameters);
  virtual void addFFTCompute(const std::string & compute_name,
                             const std::string & name,
                             InputParameters & parameters);
  virtual void addFFTIC(const std::string & compute_name,
                        const std::string & name,
                        InputParameters & parameters);

  torch::Tensor & getBuffer(const std::string & buffer_name);
  unsigned int & getDim() { return _dim; }
  const std::array<unsigned int, 3> & getGridSize() const { return _n; }
  const std::array<Real, 3> & getGridSpacing() const { return _grid_spacing; }
  const torch::Tensor & getAxis(std::size_t component) const;

protected:
  /// list of FFTBuffers (i.e. tensors)
  std::map<std::string, torch::Tensor> _fft_buffer;

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

  // compute objects
  std::vector<std::shared_ptr<FFTCompute>> _computes;

  // ice objects
  std::vector<std::shared_ptr<FFTCompute>> _ics;
};

#endif
