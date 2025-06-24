/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMMicroscopicZeroGradientBC.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannMesh.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("SwiftApp", LBMMicroscopicZeroGradientBC);

InputParameters
LBMMicroscopicZeroGradientBC::validParams()
{
  InputParameters params = LBMBoundaryCondition::validParams();
  params.addClassDescription("LBMMicroscopicZeroGradientBC object");
  return params;
}

LBMMicroscopicZeroGradientBC::LBMMicroscopicZeroGradientBC(const InputParameters & parameters)
  : LBMBoundaryCondition(parameters)
{
}

void
LBMMicroscopicZeroGradientBC::leftBoundary()
{
  _u.index_put_({1, Slice(), Slice(), Slice()}, _u.index({0, Slice(), Slice(), Slice()}));
}

void
LBMMicroscopicZeroGradientBC::rightBoundary()
{
  _u.index_put_({_grid_size[0] - 1, Slice(), Slice(), Slice()},
                _u.index({_grid_size[0] - 2, Slice(), Slice(), Slice()}));
}

void
LBMMicroscopicZeroGradientBC::computeBuffer()
{
  // do not overwrite previous
  _u = _u.clone();

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
  _lb_problem.maskedFillSolids(_u, 0);
}
