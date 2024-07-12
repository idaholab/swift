//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTMesh.h"

#include "MooseApp.h"

#include "libmesh/mesh_generation.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/unstructured_mesh.h"

registerMooseObject("SwiftApp", FFTMesh);

InputParameters
FFTMesh::validParams()
{
  InputParameters params = MooseMesh::validParams();

  MooseEnum dims("1=1 2 3");
  params.addRequiredParam<MooseEnum>("dim", dims, "The dimension of the mesh to be generated");

  params.addParam<unsigned int>("nx", 1, "Number of elements in the X direction");
  params.addParam<unsigned int>("ny", 1, "Number of elements in the Y direction");
  params.addParam<unsigned int>("nz", 1, "Number of elements in the Z direction");
  params.addParam<Real>("xmax", 1.0, "Upper X Coordinate of the generated mesh");
  params.addParam<Real>("ymax", 1.0, "Upper Y Coordinate of the generated mesh");
  params.addParam<Real>("zmax", 1.0, "Upper Z Coordinate of the generated mesh");

  params.set<bool>("allow_renumbering") = false;
  params.set<bool>("dummy_mesh") = false;
  params.suppressParameter<bool>("allow_renumbering");
  params.addClassDescription("Create a line, square, or cube mesh with uniformly spaced elements.");
  return params;
}

FFTMesh::FFTMesh(const InputParameters & parameters)
  : MooseMesh(parameters),
    _dim(getParam<MooseEnum>("dim")),
    _nx(getParam<unsigned int>("nx")),
    _ny(getParam<unsigned int>("ny")),
    _nz(getParam<unsigned int>("nz")),
    _xmax(getParam<Real>("xmax")),
    _ymax(getParam<Real>("ymax")),
    _zmax(getParam<Real>("zmax"))
{
  // All generated meshes are regular orthogonal meshes - until they get modified ;)
  _regular_orthogonal_mesh = true;

  // set unused dimensions to 1
  if (_dim <= 2)
    _nz = 1;
  if (_dim <= 1)
    _ny = 1;
  if (_dim == 0)
    _nx = 1;

  // Error check
  if (_nx == 0)
    paramError("nx", "Number of grid points in any direction must be greater than zero");
  if (_ny == 0)
    paramError("ny", "Number of grid points in any direction must be greater than zero");
  if (_nz == 0)
    paramError("nz", "Number of grid points in any direction must be greater than zero");
}

void
FFTMesh::prepared(bool state)
{
  MooseMesh::prepared(state);

  // Fall back on scanning the mesh for coordinates instead of using input parameters for queries
  if (!state)
    mooseError("FFTMesh must not be modified");
}

unsigned int
FFTMesh::getElementsInDimension(unsigned int component) const
{
  switch (component)
  {
    case 0:
      return _nx;
    case 1:
      return _ny;
    case 2:
      return _nz;
    default:
      mooseError("Invalid component");
  }
}

Real
FFTMesh::getMinInDimension(unsigned int) const
{
  return 0.0;
}

Real
FFTMesh::getMaxInDimension(unsigned int component) const
{
  switch (component)
  {
    case 0:
      return _xmax;
    case 1:
      return _dim > 1 ? _ymax : 0.0;
    case 2:
      return _dim > 2 ? _zmax : 0.0;
    default:
      mooseError("Invalid component");
  }
}

std::unique_ptr<MooseMesh>
FFTMesh::safeClone() const
{
  return _app.getFactory().copyConstruct(*this);
}

void
FFTMesh::buildMesh()
{
  auto dummy = getParam<bool>("dummy_mesh");

  // Switching on MooseEnum
  switch (_dim)
  {
    // The build_XYZ mesh generation functions take an
    // UnstructuredMesh& as the first argument, hence the dynamic_cast.
    case 1:
    {

      auto elem_type = Utility::string_to_enum<ElemType>("EDGE2");
      MeshTools::Generation::build_line(dynamic_cast<UnstructuredMesh &>(getMesh()),
                                        dummy ? 1 : _nx,
                                        0.0,
                                        _xmax,
                                        elem_type,
                                        false);
      break;
    }

    case 2:
    {
      auto elem_type = Utility::string_to_enum<ElemType>("QUAD4");
      MeshTools::Generation::build_square(dynamic_cast<UnstructuredMesh &>(getMesh()),
                                          dummy ? 1 : _nx,
                                          dummy ? 1 : _ny,
                                          0.0,
                                          _xmax,
                                          0.0,
                                          _ymax,
                                          elem_type,
                                          false);
      break;
    }

    case 3:
    {
      auto elem_type = Utility::string_to_enum<ElemType>("HEX8");
      MeshTools::Generation::build_cube(dynamic_cast<UnstructuredMesh &>(getMesh()),
                                        dummy ? 1 : _nx,
                                        dummy ? 1 : _ny,
                                        dummy ? 1 : _nz,
                                        0.0,
                                        _xmax,
                                        0.0,
                                        _ymax,
                                        0.0,
                                        _zmax,
                                        elem_type,
                                        false);
      break;
    }
  }
}
