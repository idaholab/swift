//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "MooseMesh.h"
#include "DomainInterface.h"

/**
 * Mesh generated from parameters
 */
class UniformTensorMesh : public MooseMesh
{
public:
  static InputParameters validParams();

  UniformTensorMesh(const InputParameters & parameters);
  UniformTensorMesh(const UniformTensorMesh & /* other_mesh */) = default;

  // No copy
  UniformTensorMesh & operator=(const UniformTensorMesh & other_mesh) = delete;

  virtual std::unique_ptr<MooseMesh> safeClone() const override;

  unsigned int getDim() const { return _dim; }

  virtual void buildMesh() override;
  unsigned int getElementsInDimension(unsigned int component) const;
  virtual Real getMinInDimension(unsigned int component) const override;
  virtual Real getMaxInDimension(unsigned int component) const override;
  virtual void prepared(bool state) override;

protected:
  /// The dimension of the mesh
  MooseEnum _dim;

  /// Number of elements in x, y, z direction
  unsigned int _nx, _ny, _nz;

  /// The max values for x,y,z component
  Real _xmax, _ymax, _zmax;
};
