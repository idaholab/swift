/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMFixedPressureBC.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannMesh.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("SwiftApp", LBMFixedPressureBC2D);
registerMooseObject("SwiftApp", LBMFixedPressureBC3D);

template <int dimension>
InputParameters
LBMFixedPressureBCTempl<dimension>::validParams()
{
  InputParameters params = LBMBoundaryCondition::validParams();
  params.addClassDescription("LBMFixedPressureBC object");
  params.addRequiredParam<TensorInputBufferName>(
      "f", "Input buffer distribution function");
  params.addRequiredParam<double>("value", "Fixed input value");
  return params;
}

template <int dimension>
LBMFixedPressureBCTempl<dimension>::LBMFixedPressureBCTempl(const InputParameters & parameters)
    : LBMBoundaryCondition(parameters),
    _f(getInputBuffer("f")),
    _grid_size(_lb_problem.getGridSize()),
    _value(getParam<Real>("value"))
{
}

template <>
void
LBMFixedPressureBCTempl<2>::topBoundary()
{
  // There is not top boundary in 2D
}

template <>
void
LBMFixedPressureBCTempl<3>::topBoundary()
{
  // TBD
}

template <>
void
LBMFixedPressureBCTempl<2>::bottomBoundary()
{
  // There is not bottom boundary in 2D
}

template <>
void
LBMFixedPressureBCTempl<3>::bottomBoundary()
{
  // TBD
}

template <>
void
LBMFixedPressureBCTempl<2>::leftBoundary()
{
  torch::Tensor velocity = 1.0 - (_f.index({0, Slice(), Slice(), 0}) + 
                        _f.index({0, Slice(), Slice(), 2}) + 
                        _f.index({0, Slice(), Slice(), 4}) + 
                        2 * (_f.index({0, Slice(), Slice(), 3}) + 
                        _f.index({0, Slice(), Slice(), 6}) + 
                        _f.index({0, Slice(), Slice(), 7}))) / _value;
  
  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._left[0]];
  _u.index_put_({0, Slice(), Slice(), _stencil._left[0]}, _f.index({0, Slice(), Slice(), opposite_dir}) + 2.0/3.0 * _value * velocity);

  // other directions
  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._left[i]];
    _u.index_put_({0, Slice(), Slice(), _stencil._left[i]}, _f.index({0, Slice(), Slice(), opposite_dir}) - 0.5 * _stencil._ey[_stencil._left[i]] * 
                                                          (_f.index({0, Slice(), Slice(), 2}) - _f.index({0, Slice(), Slice(), 4})) + 
                                                          1.0/6.0 * _value * velocity );
  }
}

template <>
void
LBMFixedPressureBCTempl<3>::leftBoundary()
{
  // TBD
}

template <>
void
LBMFixedPressureBCTempl<2>::rightBoundary()
{ 
  torch::Tensor velocity = (_f.index({_grid_size[0] - 1, Slice(), Slice(), 0}) + 
                        _f.index({_grid_size[0] - 1, Slice(), Slice(), 2}) + 
                        _f.index({_grid_size[0] - 1, Slice(), Slice(), 4}) + 
                        2 * (_f.index({_grid_size[0] - 1, Slice(), Slice(), 1}) + 
                        _f.index({_grid_size[0] - 1, Slice(), Slice(), 5}) + 
                        _f.index({_grid_size[0] - 1, Slice(), Slice(), 8}))) / _value - 1.0;
  
  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._left[0]];
  _u.index_put_({_grid_size[0] - 1, Slice(), Slice(), opposite_dir}, _f.index({_grid_size[0] - 1, Slice(), Slice(), _stencil._left[0]}) - 2.0/3.0 * _value * velocity);

  // other directions
  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._left[i]];
    _u.index_put_({_grid_size[0] - 1, Slice(), Slice(), opposite_dir}, _f.index({_grid_size[0] - 1, Slice(), Slice(), _stencil._left[i]}) + 0.5 * _stencil._ey[opposite_dir] * 
                                                          (_f.index({_grid_size[0] - 1, Slice(), Slice(), 4}) - _f.index({_grid_size[0] - 1, Slice(), Slice(), 2})) - 
                                                          1.0/6.0 * _value * velocity );
  }
}

template <>
void
LBMFixedPressureBCTempl<3>::rightBoundary()
{
  // TBD
}

template <>
void
LBMFixedPressureBCTempl<2>::frontBoundary()
{
  torch::Tensor velocity = 1.0 - (_f.index({Slice(), 0, Slice(), 0}) + 
                        _f.index({Slice(), 0, Slice(), 1}) + 
                        _f.index({Slice(), 0, Slice(), 3}) + 
                        2 * (_f.index({Slice(), 0, Slice(), 4}) + 
                        _f.index({Slice(), 0, Slice(), 7}) + 
                        _f.index({Slice(), 0, Slice(), 8}))) / _value;
  
  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._front[0]];
  _u.index_put_({Slice(), 0, Slice(), _stencil._front[0]}, _f.index({Slice(), 0, Slice(), opposite_dir}) + 2.0/3.0 * _value * velocity);

  // other directions
  for (unsigned int i = 1; i < _stencil._front.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._front[i]];
    _u.index_put_({Slice(), 0, Slice(), _stencil._front[i]}, _f.index({Slice(), 0, Slice(), opposite_dir}) - 0.5 * _stencil._ex[_stencil._front[i]] * 
                                                          (_f.index({Slice(), 0, Slice(), 1}) - _f.index({Slice(), 0, Slice(), 3})) + 
                                                          1.0/6.0 * _value * velocity );
  }
}

template <>
void
LBMFixedPressureBCTempl<3>::frontBoundary()
{
  // TBD
}

template <>
void
LBMFixedPressureBCTempl<2>::backBoundary()
{
  torch::Tensor velocity = (_f.index({Slice(), _grid_size[1] - 1, Slice(), 0}) + 
                        _f.index({Slice(), _grid_size[1] - 1, Slice(), 1}) + 
                        _f.index({Slice(), _grid_size[1] - 1, Slice(), 3}) + 
                        2 * (_f.index({Slice(), _grid_size[1] - 1, Slice(), 2}) + 
                        _f.index({Slice(), _grid_size[1] - 1, Slice(), 5}) + 
                        _f.index({Slice(), _grid_size[1] - 1, Slice(), 6}))) / _value - 1.0;
  
  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._front[0]];
  _u.index_put_({Slice(), _grid_size[1] - 1, Slice(), opposite_dir}, _f.index({Slice(), _grid_size[1] - 1, Slice(), _stencil._front[0]}) - 2.0/3.0 * _value * velocity);

  // other directions
  for (unsigned int i = 1; i < _stencil._front.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._front[i]];
    _u.index_put_({Slice(), _grid_size[1] - 1, Slice(), opposite_dir}, _f.index({Slice(), _grid_size[1] - 1, Slice(), _stencil._front[i]}) + 0.5 * _stencil._ey[opposite_dir] * 
                                                          (_f.index({Slice(), _grid_size[1] - 1, Slice(), 4}) - _f.index({Slice(), _grid_size[1] - 1, Slice(), 2})) - 
                                                          1.0/6.0 * _value * velocity );
  }
}

template <>
void
LBMFixedPressureBCTempl<3>::backBoundary()
{
  // TBD
}

template <int dimension>
void
LBMFixedPressureBCTempl<dimension>::computeBuffer()
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
  _lb_problem.maskedFillSolids(_u, 0);
}

template class LBMFixedPressureBCTempl<2>;
template class LBMFixedPressureBCTempl<3>;
