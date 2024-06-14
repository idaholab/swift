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
  virtual void addFFTCompute(const std::string & compute_name, InputParameters & parameters);

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
  std::vector<torch::Tensor> _axis;

  /// reciprocal space axes
  std::vector<torch::Tensor> _reciprocal_axis;
};

#endif
