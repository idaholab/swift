/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "Action.h"
#include "SwiftUtils.h"
#include <vector>
#include <string>
#include <array>
#include <initializer_list>

#include <torch/torch.h>

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
  const std::array<int64_t, 3> & getReciprocalGridSize() const { return _n_reciprocal_global; }
  const std::array<int64_t, 3> & getLocalGridSize() const { return _n_local; }
  const std::array<int64_t, 3> & getLocalReciprocalGridSize() const { return _n_reciprocal_local; }
  const Real & getVolume() const { return _volume_global; }
  const RealVectorValue & getDomainMin() const { return _min_global; }
  const RealVectorValue & getDomainMax() const { return _max_global; }
  const RealVectorValue & getGridSpacing() const { return _grid_spacing; }
  const torch::Tensor & getAxis(std::size_t component) const;
  const torch::Tensor & getReciprocalAxis(std::size_t component) const;
  const torch::Tensor & getKSquare() const { return _k2; }

  /// get the maximum spatial frequency
  const RealVectorValue & getMaxK() const { return _max_k; }

  /// get the shape of the local domain
  const torch::IntArrayRef & getShape() const { return _shape; }
  const torch::IntArrayRef & getReciprocalShape() const { return _reciprocal_shape; }

  torch::Tensor fft(const torch::Tensor & t) const;
  torch::Tensor ifft(const torch::Tensor & t) const;

  torch::Tensor emptyReal(std::initializer_list<int64_t> extra_dims) const;
  torch::Tensor emptyReciprocal(std::initializer_list<int64_t> extra_dims) const;

  /// align a 1d tensor in a specific dimension
  torch::Tensor align(torch::Tensor t, unsigned int dim) const;

protected:
  void gridChanged();

  void partitionSerial();
  void partitionSlabs();
  void partitionPencils();

  torch::Tensor fftSerial(const torch::Tensor & t) const;
  torch::Tensor fftSlab(const torch::Tensor & t) const;
  torch::Tensor fftPencil(const torch::Tensor & t) const;

  template <bool is_real>
  torch::Tensor cosineTransform(const torch::Tensor & t, int64_t axis) const;

  template <typename T>
  std::vector<int64_t> partitionHepler(int64_t total, const std::vector<T> & weights);

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

  /// global number of grid points in real space
  std::array<int64_t, 3> _n_reciprocal_global;

  /// local number of grid points in real space
  std::array<int64_t, 3> _n_local;

  /// local number of grid points in real space
  std::array<int64_t, 3> _n_reciprocal_local;

  /// local begin/end indixes along each direction for slabs/pencils
  std::array<std::vector<int64_t>, 3> _local_begin;
  std::array<std::vector<int64_t>, 3> _local_end;
  std::array<std::vector<int64_t>, 3> _n_local_all;

  ///@{ global domain length in each dimension
  const RealVectorValue _min_global;
  const RealVectorValue _max_global;
  ///@}

  /// Volume of the simulation domain in real space
  Real _volume_global;

  const enum class MeshMode { SWIFT_DUMMY, SWIFT_DOMAIN, SWIFT_MANUAL } _mesh_mode;

  /// grid spacing
  RealVectorValue _grid_spacing;

  /// real space axes
  std::array<torch::Tensor, 3> _global_axis;
  std::array<torch::Tensor, 3> _local_axis;

  /// reciprocal space axes
  std::array<torch::Tensor, 3> _global_reciprocal_axis;
  std::array<torch::Tensor, 3> _local_reciprocal_axis;

  /// k-square
  torch::Tensor _k2;

  /// largest frequency along each axis
  RealVectorValue _max_k;

  /// domain shape
  torch::IntArrayRef _shape;
  torch::IntArrayRef _reciprocal_shape;

  /// MPI rank
  unsigned int _rank;

  /// number of MPI ranks
  unsigned int _n_rank;

  /// send tensors
  mutable std::vector<torch::Tensor> _send_tensor;
  /// receive buffer
  mutable std::vector<std::vector<double>> _recv_data;
  /// receive tensors
  mutable std::vector<torch::Tensor> _recv_tensor;
};

template <typename T>
std::vector<int64_t>
DomainAction::partitionHepler(int64_t total, const std::vector<T> & weights)
{
  std::vector<int64_t> ns;

  T remaining_total_weight = 0;
  for (const auto w : weights)
    remaining_total_weight += w;

  for (const auto w : weights)
  {
    if (remaining_total_weight == 0)
      mooseError("Internal partitioning error. remaining_total_weight ",
                 remaining_total_weight,
                 " == 0 ",
                 _rank);

    // assign at least one layer
    const auto n = std::max((total * w) / remaining_total_weight, int64_t(1));
    ns.push_back(n);

    remaining_total_weight -= w;

    if (total < n)
      mooseError("Internal partitioning error.");

    total -= n;
  }

  // add remainsder to last slice
  ns.back() += total;
  return ns;
}

// See Makhoul 2003 (DOI: 10.1109/TASSP.1980.1163351)
template <bool is_real>
torch::Tensor
DomainAction::cosineTransform(const torch::Tensor & t, int64_t axis) const
{
  // size along the axis
  // const auto l = t.sizes()[axis];

  // mirror tensor and stack onto itself (with one layer removed)
  auto t_flip = torch::flip(t, {axis});

  // stack tensor along axis
  auto t_stacked = torch::stack({t, t_flip}, axis);

  // perform 1D FFT along the selected axis and slice in the reciprocal domain
  torch::Tensor t_bar;
  if constexpr (is_real)
    t_bar = torch::fft::rfft(t_stacked, -1, axis);
  else
    t_bar = torch::fft::fft(t_stacked, -1, axis);

  mooseError("Not implemented!");
  // return t_bar;
}
