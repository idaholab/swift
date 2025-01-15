/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMBoundaryCondition.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"
#include "LatticeBoltzmannMesh.h"

InputParameters
LBMBoundaryCondition::validParams()
{
    InputParameters params = LatticeBoltzmannOperator::validParams();
    MooseEnum boundary("top bottom left right front back wall");
    params.addRequiredParam<MooseEnum>("boundary", boundary, "Edges/Faces where boundary condition is applied.");
    params.addClassDescription("LBMBoundaryCondition object.");
    return params;
}

LBMBoundaryCondition::LBMBoundaryCondition(const InputParameters & parameters)
    : LatticeBoltzmannOperator(parameters),
    _boundary(getParam<MooseEnum>("boundary").getEnum<Boundary>())
{
  /**
   * Nodes that are adjacent to boundary will be set to 2, this will later be used in determining
   * the nodes for bounce-back
   * This will be achieved by shifting the mesh around in streaming directions and finding where
   * boundary hit happens
   */
  if (_mesh.isMeshDatFile() || _mesh.isMeshVTKFile())
  {
    torch::Tensor new_mesh = _mesh.getBinaryMesh().clone();
    for (int64_t ic = 1; ic < _stencil._q; ic++)
    {
      int64_t ex = _stencil._ex[ic].item<int64_t>();
      int64_t ey = _stencil._ey[ic].item<int64_t>();
      int64_t ez = _stencil._ez[ic].item<int64_t>();
      torch::Tensor shifted_mesh = torch::roll(new_mesh, {ex, ey, ez}, {0, 1, 2});
      torch::Tensor adjacent_to_boundary = (shifted_mesh == 0) & (new_mesh == 1);
      new_mesh.masked_fill_(adjacent_to_boundary, 2);
    }
    // Deep copy new mesh
    // MooseTensor::printField(new_mesh, 1, 0);
    _mesh.setBinaryMesh(new_mesh);
  }
}

void
LBMBoundaryCondition::computeBuffer()
{ 
  switch (_boundary)
  {
    case Boundary::top:
      topBoundary();
      break;
    case Boundary::bottom:
      bottomBoundary();
      break;
    case Boundary::left:
      leftBoundary();
      break;
    case Boundary::right:
      rightBoundary();
      break;
    case Boundary::front:
      frontBoundary();
      break;
    case Boundary::back:
      backBoundary();
      break;
    case Boundary::wall:
      wallBoundary();
      break;
    default:
      mooseError("Undefined boundary names");
  }
}
