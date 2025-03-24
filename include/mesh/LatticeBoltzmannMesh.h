/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "UniformTensorMesh.h"
#include "SwiftUtils.h"

/**
 * Lattice Boltzmann Mesh generated from parameters
 */
class LatticeBoltzmannMesh : public UniformTensorMesh
{
public:
  static InputParameters validParams();

  LatticeBoltzmannMesh(const InputParameters & parameters);

  void buildMesh() override;

  void loadMeshFromDatFile();
  void loadMeshFromVTKFile(const std::string&, torch::Tensor&, torch::Tensor&, torch::Tensor&);
  const torch::Tensor & getBinaryMesh() const {return _binary_mesh;};
  const torch::Tensor & getKn() const {return _Knudsen_number;};
  const torch::Tensor & getPoreSize() const {return _local_pore_size;};
  const bool & isMeshDatFile() const {return _load_mesh_from_dat;};
  const bool & isMeshVTKFile() const {return _load_mesh_from_vtk;};
  void setBinaryMesh(torch::Tensor & new_mesh) {_binary_mesh = new_mesh.clone();};

protected:
  torch::Tensor _binary_mesh;
  torch::Tensor _local_pore_size;
  torch::Tensor _Knudsen_number;
  bool _load_mesh_from_dat;
  bool _load_mesh_from_vtk;
  std::string _mesh_file;
};
