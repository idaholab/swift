//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "CoordinatePlaneNodalContactBC.h"

registerMooseObject("MooseApp", CoordinatePlaneNodalContactBC);

InputParameters
CoordinatePlaneNodalContactBC::validParams()
{
  InputParameters params = DirichletBC::validParams();
  params.addClassDescription("Imposes a contact enforcement with the negative half-space in the "
                             "direction of the coupled displacement variable.");
  params.addRequiredCoupledVar("displacements",
                               "The displacement components. These are only used to determine the "
                               "axis from the coupled variable.");
  MooseEnum obstacle("NEGATIVE POSITIVE", "NEGATIVE");
  params.addParam<MooseEnum>(
      "obstacle", obstacle, "Which half-space is the obstacle for contact enforcement?");
  params.set<bool>("use_displaced_mesh") = false;
  params.set<Real>("value") = 0.0;
  params.suppressParameter<bool>("use_displaced_mesh");

  return params;
}

CoordinatePlaneNodalContactBC::CoordinatePlaneNodalContactBC(const InputParameters & parameters)
  : DirichletBC(parameters), _negative(getParam<MooseEnum>("obstacle") == "NEGATIVE")
{
  auto get_component = [this]()
  {
    for (const auto i : make_range(coupledComponents("displacements")))
      if (getVar("displacements", i)->number() == _var.number())
        return i;
    paramError("displacements",
               "The coupled variable '",
               _var.name(),
               "' must be among the coupled displacements.");
  };
  _component = get_component();
}

Real
CoordinatePlaneNodalContactBC::computeQpValue()
{
  return _value - (*_current_node)(_component);
}

bool
CoordinatePlaneNodalContactBC::shouldApply()
{
  // is the node penetrating?
  const auto gap = _negative ? (*_current_node)(_component) + _u[0] - _value
                             : _value - ((*_current_node)(_component) + _u[0]);

  if (gap > libMesh::TOLERANCE)
    return false;

  // get current nodal force
  const auto residual_tag = _sys.residualVectorTag();
  if (_sys.hasVector(residual_tag))
  {
    const auto force = _sys.getVector(residual_tag)(_var.nodalDofIndex());

    // do not enforce BC if we are pointing away from the wall
    if (_negative ? force > libMesh::TOLERANCE : force < libMesh::TOLERANCE)
      return false;
  }
  else
    return false;

  // else enforce
  return true;
}
