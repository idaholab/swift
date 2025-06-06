/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMBounceBack.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannMesh.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("SwiftApp", LBMBounceBack);

InputParameters
LBMBounceBack::validParams()
{
  InputParameters params = LBMBoundaryCondition::validParams();
  params.addClassDescription("LBMBounceBack object");
  params.addRequiredParam<TensorInputBufferName>(
      "f_old", "Old state distribution function");
  return params;
}

LBMBounceBack::LBMBounceBack(const InputParameters & parameters)
    : LBMBoundaryCondition(parameters),
    _f_old(_lb_problem.getBufferOld(getParam<TensorInputBufferName>("f_old"), 1)),
    _grid_size(_lb_problem.getGridSize())
{
}

void
LBMBounceBack::topBoundary()
{
  for (unsigned int i = 0; i < _stencil._bottom.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._bottom[i]];
    _u.index_put_({Slice(), Slice(), _grid_size[2]-1, opposite_dir}, _f_old[0].index({Slice(), Slice(), _grid_size[2]-1, _stencil._bottom[i]}));
  }
}

void
LBMBounceBack::bottomBoundary()
{
  for (unsigned int i = 0; i < _stencil._bottom.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._bottom[i]];
    _u.index_put_({Slice(), Slice(), 0, _stencil._bottom[i]}, _f_old[0].index({Slice(), Slice(), 0, opposite_dir}));
  }
}

void
LBMBounceBack::leftBoundary()
{
  for (unsigned int i = 0; i < _stencil._left.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._left[i]];
    _u.index_put_({0, Slice(), Slice(), _stencil._left[i]}, _f_old[0].index({0, Slice(), Slice(), opposite_dir}));
  }
}

void
LBMBounceBack::rightBoundary()
{
  for (unsigned int i = 0; i < _stencil._left.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._left[i]];
    _u.index_put_({_grid_size[0] - 1, Slice(), Slice(), opposite_dir}, _f_old[0].index({_grid_size[0] - 1, Slice(), Slice(), _stencil._left[i]}));
  }
}

void
LBMBounceBack::frontBoundary()
{
  for (unsigned int i = 0; i < _stencil._front.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._front[i]];
    _u.index_put_({Slice(), 0, Slice(), _stencil._front[i]}, _f_old[0].index({Slice(), 0, Slice(), opposite_dir}));
  }
}

void
LBMBounceBack::backBoundary()
{
  for (unsigned int i = 0; i < _stencil._front.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._front[i]];
    _u.index_put_({Slice(), _grid_size[1] - 1, Slice(), opposite_dir}, _f_old[0].index({Slice(), _grid_size[1] - 1, Slice(), _stencil._front[i]}));
  }
}

void LBMBounceBack::wallBoundary()
{
  // std::cout<<_boundary_indices<<std::endl;
  // build boundary mask in the begining of simulation
  if (_lb_problem.getTotalSteps() == 0)
  {
    LBMBoundaryCondition::buildBoundaryIndices();
  }

  // bounce-back
  _u.index_put_({_boundary_indices.index({Slice(), 0}), 
                          _boundary_indices.index({Slice(), 1}), 
                          _boundary_indices.index({Slice(), 2}),
                          _boundary_indices.index({Slice(), 3})}, 
                          
      _f_old[0].index({_boundary_indices.index({Slice(), 0}), 
                                    _boundary_indices.index({Slice(), 1}), 
                                    _boundary_indices.index({Slice(), 2}),
                                    _stencil._op.index_select(0, _boundary_indices.index({Slice(), 3}))}));
}

void
LBMBounceBack::computeBuffer()
{
  const auto n_old = _f_old.size();
  if (n_old != 0)
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
  }
  _lb_problem.maskedFillSolids(_u, 0);
}
