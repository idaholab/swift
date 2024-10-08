
//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "DomainMeshGenerator.h"

registerMooseObject("SwiftApp", DomainMeshGenerator);

InputParameters
DomainMeshGenerator::validParams()
{
  InputParameters params = GeneratedMeshGenerator::validParams();
  return params;
}

DomainMeshGenerator::DomainMeshGenerator(const InputParameters & parameters)
  : GeneratedMeshGenerator(parameters)
{
}

std::unique_ptr<MeshBase>
DomainMeshGenerator::generate()
{
  auto mesh = GeneratedMeshGenerator::generate();
  mesh->allow_renumbering(false);
  return mesh;
}
