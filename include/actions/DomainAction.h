//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "Action.h"
#include <vector>
#include <string>
#include <array>

#include "torch/torch.h"

/**
 * This class adds an TensorBuffer object.
 * The TensorBuffer is a structured grid object using a libtorch tensor to store data
 * in real space. A reciprocal space representation is automatically created on demand.
 */
class DomainAction : public Action
{
public:
  static InputParameters validParams();

  DomainAction(const InputParameters & parameters);

  virtual void act() override;

  const unsigned int & getDim() const { return _dim; }
  const std::array<int64_t, 3> & getGridSize() const { return _n_global; }
  const std::array<Real, 3> & getDomainSize() const { return _max_global; }
  const std::array<Real, 3> & getGridSpacing() const { return _grid_spacing; }
  const torch::Tensor & getAxis(std::size_t component) const;
  const torch::Tensor & getReciprocalAxis(std::size_t component) const;
  const torch::Tensor & getKSquare() const { return _k2; }
  const torch::IntArrayRef  & getShape() const { return _shape; }

  torch::Tensor fft(torch::Tensor t) const;
  torch::Tensor ifft(torch::Tensor t) const;

  /// align a 1d tensor in a specific dimension
  torch::Tensor align(torch::Tensor t, unsigned int dim) const;

protected:
  void gridChanged();

  void partitionSerial();
  void partitionSlabs();
  void partitionPencils();

  /// device names to be used on the nodes
  const std::vector<std::string> _device_names;

  /// device weights to be used on the nodes
  std::vector<unsigned int> _device_weights;

  /// parallelization mode
  const enum class ParallelMode { NONE, FFT_SLAB, FFT_PENCIL } _parallel_mode;

  /// host local ranks of all procs
  std::vector<unsigned int> _local_ranks;
  std::vector<unsigned int> _local_weights;

  /// The dimension of the mesh
  const unsigned int _dim;

  /// global number of grid points in real space
  const std::array<int64_t, 3> _n_global;

  /// local number of grid points in real space
  std::array<int64_t, 3> _n_local;

  /// local begin/end indixes along each direction for slabs/pencils
  std::array<std::vector<int64_t>, 3> _local_begin;
  std::array<std::vector<int64_t>, 3> _local_end;

  /// global domain length in each dimension
  const std::array<Real, 3> _max_global;

  const enum class MeshMode { DUMMY, DOMAIN, MANUAL} _mesh_mode;

  /// grid spacing
  std::array<Real, 3> _grid_spacing;

  /// real space axes
  std::array<torch::Tensor, 3> _global_axis;
  std::array<torch::Tensor, 3> _local_axis;

  /// reciprocal space axes
  std::array<torch::Tensor, 3> _global_reciprocal_axis;
  std::array<torch::Tensor, 3> _local_reciprocal_axis;

  /// k-square
  torch::Tensor _k2;

  /// domain shape
  torch::IntArrayRef _shape;
};
