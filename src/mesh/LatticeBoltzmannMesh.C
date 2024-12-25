/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LatticeBoltzmannMesh.h"

registerMooseObject("SwiftApp", LatticeBoltzmannMesh);

InputParameters
LatticeBoltzmannMesh::validParams()
{
  InputParameters params = UniformTensorMesh::validParams();

  params.addParam<bool>("load_mesh_from_dat", false, "Load mesh from dat file");
  params.addParam<bool>("load_mesh_from_vtk", false, "Load mesh from VTK file");
  params.addParam<std::string>("mesh_file", "", "Mesh file name");
  
  params.addClassDescription("Create mesh file for LBM problems.");

  return params;
}

LatticeBoltzmannMesh::LatticeBoltzmannMesh(const InputParameters & parameters)
  : UniformTensorMesh(parameters),
  _load_mesh_from_dat(getParam<bool>("load_mesh_from_dat")),
  _load_mesh_from_vtk(getParam<bool>("load_mesh_from_vtk")),
  _mesh_file(getParam<std::string>("mesh_file"))
{
}

void
LatticeBoltzmannMesh::buildMesh()
{
  // call base class buildMesh
  UniformTensorMesh::buildMesh();

  if (_load_mesh_from_dat)
  {
    _binary_mesh = torch::ones({_nx, _ny, _nz}, MooseTensor::intTensorOptions());
    loadMeshFromDatFile();
  }
  else if (_load_mesh_from_vtk)
  {
    _binary_mesh = torch::ones({_nx, _ny, _nz}, MooseTensor::intTensorOptions());
    _local_pore_size = torch::ones({_nx, _ny, _nz}, MooseTensor::floatTensorOptions());
    _Knudsen_number = torch::ones({_nx, _ny, _nz}, MooseTensor::floatTensorOptions());
    loadMeshFromVTKFile();
  }
    
}

void
LatticeBoltzmannMesh::loadMeshFromDatFile()
{
  auto dummy = getParam<bool>("dummy_mesh");
  if (dummy)
    mooseError("Cannot load mesh from file when dummy mesh is enabled");

  _console<<COLOR_WHITE<<"Loading Binary Mesh From Dat File\n";
  
  std::ifstream file(_mesh_file);
  if (!file.is_open())
    mooseError("Cannot open file " + _mesh_file);

  // read mesh into standart vector
  std::vector<std::vector<int>> matrixData;
  std::string line;
  while (std::getline(file, line))
  {
    std::istringstream iss(line);
    int num;
    std::vector<int> row;
    while (iss >> num)
        row.push_back(num);
    matrixData.push_back(row);
  }
  file.close();

  // reshape and write into torch tensor
  for(int64_t i = 0; i <_binary_mesh.size(2); i++)
    for(int64_t j = 0; j < _binary_mesh.size(1); j++)
        for(int64_t k = 0; k < _binary_mesh.size(0); k++)
          _binary_mesh.index_put_({k, j, i}, matrixData[i * _binary_mesh.size(2) + j][k]);
}

void
LatticeBoltzmannMesh::loadMeshFromVTKFile()
{
  if (_dim != 2)
    mooseError("VTK mesh reader is only supported for 2D cases");

  _console<<COLOR_WHITE<<"Loading Binary Mesh From VTK File\n";

  std::vector<int> dims = {static_cast<int>(_nx),
                          static_cast<int>(_ny), static_cast<int>(_nz)}; // fixes narrowing conversion warning

  #ifdef SWIFT_HAVE_VTK
   MooseTensor::read2DStructuredLBMMeshFromVTK(_mesh_file, _binary_mesh, _local_pore_size, _Knudsen_number, dims);
  #else
    mooseError("VTK not enabled");
  #endif
}
