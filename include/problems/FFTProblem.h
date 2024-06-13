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
  unsigned int _nx, _ny, _nz;

  /// The max values for x,y,z component
  Real _xmax, _ymax, _zmax;

  /// grid spacing
  Real _dx, _dy, _dz;

  /// domain shape
  std::vector<long int> _shape_storage;
  torch::IntArrayRef _shape;
};

#endif
