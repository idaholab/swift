/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMFixedZerothOrderBC.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannMesh.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("SwiftApp", LBMFixedZerothOrderBC9Q);
registerMooseObject("SwiftApp", LBMFixedZerothOrderBC19Q);
registerMooseObject("SwiftApp", LBMFixedZerothOrderBC27Q);

template <int dimension>
InputParameters
LBMFixedZerothOrderBCTempl<dimension>::validParams()
{
  InputParameters params = LBMBoundaryCondition::validParams();
  params.addClassDescription("LBMFixedZerothOrderBC object");
  params.addRequiredParam<TensorInputBufferName>("f", "Input buffer distribution function");
  params.addRequiredParam<std::string>("value", "Fixed input value");
  return params;
}

template <int dimension>
LBMFixedZerothOrderBCTempl<dimension>::LBMFixedZerothOrderBCTempl(
    const InputParameters & parameters)
  : LBMBoundaryCondition(parameters),
    _f(getInputBufferByName(getParam<TensorInputBufferName>("f"))),
    _grid_size(_lb_problem.getGridSize()),
    _value(_lb_problem.getConstant<Real>(getParam<std::string>("value")))
{
}

template <>
void
LBMFixedZerothOrderBCTempl<9>::frontBoundary()
{
  // There is no front boundary in 2D
  mooseError("There is no front boundary in 2 dimensions.");
}

template <>
void
LBMFixedZerothOrderBCTempl<19>::frontBoundary()
{
  // TBD
}

template <>
void
LBMFixedZerothOrderBCTempl<27>::frontBoundary()
{
  // TBD
}

template <>
void
LBMFixedZerothOrderBCTempl<9>::backBoundary()
{
  // There is no back boundary in 2D
  mooseError("There is no back boundary in 2 dimensions.");
}

template <>
void
LBMFixedZerothOrderBCTempl<19>::backBoundary()
{
  // TBD
}

template <>
void
LBMFixedZerothOrderBCTempl<27>::backBoundary()
{
  // TBD
}

template <>
void
LBMFixedZerothOrderBCTempl<9>::leftBoundary()
{
  torch::Tensor velocity =
      1.0 - (_f.index({0, Slice(), Slice(), 0}) + _f.index({0, Slice(), Slice(), 2}) +
             _f.index({0, Slice(), Slice(), 4}) +
             2 * (_f.index({0, Slice(), Slice(), 3}) + _f.index({0, Slice(), Slice(), 6}) +
                  _f.index({0, Slice(), Slice(), 7}))) /
                _value;

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._left[0]];
  _u.index_put_({0, Slice(), Slice(), _stencil._left[0]},
                _f.index({0, Slice(), Slice(), opposite_dir}) + 2.0 / 3.0 * _value * velocity);

  // other directions
  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._left[i]];
    _u.index_put_(
        {0, Slice(), Slice(), _stencil._left[i]},
        _f.index({0, Slice(), Slice(), opposite_dir}) -
            0.5 * _stencil._ey[_stencil._left[i]] *
                (_f.index({0, Slice(), Slice(), 2}) - _f.index({0, Slice(), Slice(), 4})) +
            1.0 / 6.0 * _value * velocity);
  }
}

template <>
void
LBMFixedZerothOrderBCTempl<19>::leftBoundary()
{
  // TBD
}

template <>
void
LBMFixedZerothOrderBCTempl<27>::leftBoundary()
{
  torch::Tensor velocity = 1.0 - (_f.index({0, Slice(), Slice(), -_stencil._neutral_x}) +
                                  2 * _f.index({0, Slice(), Slice(), _stencil._right})) /
                                     _value;

  _u.index_put_({0, Slice(), Slice(), _stencil._left[0]},
                _f.index({0, Slice(), Slice(), _stencil._right[0]}) +
                    2.0 * _stencil._weights[_stencil._left[0]] / _lb_problem._cs2 * _value *
                        velocity);

  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    _u.index_put_(
        {0, Slice(), Slice(), _stencil._left[i]},
        _f.index({0, Slice(), Slice(), _stencil._right[i]}) +
            2.0 * _stencil._weights[_stencil._left[i]] / _lb_problem._cs2 * _value * velocity -
            0.5 * _stencil._ey[_stencil._left[i]] *
                (torch::sum(_f.index({0, Slice(), Slice(), _stencil._neutral_x_pos_y}), 3) -
                 torch::sum(_f.index({0, Slice(), Slice(), _stencil._neutral_x_neg_y}), 3)) -
            0.5 * _stencil._ez[_stencil._left[i]] *
                (torch::sum(_f.index({0, Slice(), Slice(), _stencil._neutral_x_pos_z}), 3) -
                 torch::sum(_f.index({0, Slice(), Slice(), _stencil._neutral_x_neg_z}), 3)));
  }
}

template <>
void
LBMFixedZerothOrderBCTempl<9>::rightBoundary()
{
  torch::Tensor velocity = (_f.index({_grid_size[0] - 1, Slice(), Slice(), 0}) +
                            _f.index({_grid_size[0] - 1, Slice(), Slice(), 2}) +
                            _f.index({_grid_size[0] - 1, Slice(), Slice(), 4}) +
                            2 * (_f.index({_grid_size[0] - 1, Slice(), Slice(), 1}) +
                                 _f.index({_grid_size[0] - 1, Slice(), Slice(), 5}) +
                                 _f.index({_grid_size[0] - 1, Slice(), Slice(), 8}))) /
                               _value -
                           1.0;

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._left[0]];
  _u.index_put_({_grid_size[0] - 1, Slice(), Slice(), opposite_dir},
                _f.index({_grid_size[0] - 1, Slice(), Slice(), _stencil._left[0]}) -
                    2.0 / 3.0 * _value * velocity);

  // other directions
  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._left[i]];
    _u.index_put_({_grid_size[0] - 1, Slice(), Slice(), opposite_dir},
                  _f.index({_grid_size[0] - 1, Slice(), Slice(), _stencil._left[i]}) +
                      0.5 * _stencil._ey[opposite_dir] *
                          (_f.index({_grid_size[0] - 1, Slice(), Slice(), 4}) -
                           _f.index({_grid_size[0] - 1, Slice(), Slice(), 2})) -
                      1.0 / 6.0 * _value * velocity);
  }
}

template <>
void
LBMFixedZerothOrderBCTempl<19>::rightBoundary()
{
  // TBD
}

template <>
void
LBMFixedZerothOrderBCTempl<27>::rightBoundary()
{
  torch::Tensor velocity = (_f.index({0, Slice(), Slice(), -_stencil._neutral_x}) +
                            2 * _f.index({0, Slice(), Slice(), _stencil._right})) /
                               _value -
                           1.0;

  _u.index_put_({0, Slice(), Slice(), _stencil._right[0]},
                _f.index({0, Slice(), Slice(), _stencil._left[0]}) -
                    2.0 * _stencil._weights[_stencil._right[0]] / _lb_problem._cs2 * _value *
                        velocity);

  for (unsigned int i = 1; i < _stencil._right.size(0); i++)
  {
    _u.index_put_(
        {0, Slice(), Slice(), _stencil._right[i]},
        _f.index({0, Slice(), Slice(), _stencil._left[i]}) -
            2.0 * _stencil._weights[_stencil._right[i]] / _lb_problem._cs2 * _value * velocity +
            0.5 * _stencil._ey[_stencil._right[i]] *
                (torch::sum(_f.index({0, Slice(), Slice(), _stencil._neutral_x_pos_y}), 3) -
                 torch::sum(_f.index({0, Slice(), Slice(), _stencil._neutral_x_neg_y}), 3)) +
            0.5 * _stencil._ez[_stencil._right[i]] *
                (torch::sum(_f.index({0, Slice(), Slice(), _stencil._neutral_x_pos_z}), 3) -
                 torch::sum(_f.index({0, Slice(), Slice(), _stencil._neutral_x_neg_z}), 3)));
  }
}

template <>
void
LBMFixedZerothOrderBCTempl<9>::bottomBoundary()
{
  torch::Tensor velocity =
      1.0 - (_f.index({Slice(), 0, Slice(), 0}) + _f.index({Slice(), 0, Slice(), 1}) +
             _f.index({Slice(), 0, Slice(), 3}) +
             2 * (_f.index({Slice(), 0, Slice(), 4}) + _f.index({Slice(), 0, Slice(), 7}) +
                  _f.index({Slice(), 0, Slice(), 8}))) /
                _value;

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._bottom[0]];
  _u.index_put_({Slice(), 0, Slice(), _stencil._bottom[0]},
                _f.index({Slice(), 0, Slice(), opposite_dir}) + 2.0 / 3.0 * _value * velocity);

  // other directions
  for (unsigned int i = 1; i < _stencil._bottom.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._bottom[i]];
    _u.index_put_(
        {Slice(), 0, Slice(), _stencil._bottom[i]},
        _f.index({Slice(), 0, Slice(), opposite_dir}) -
            0.5 * _stencil._ex[_stencil._bottom[i]] *
                (_f.index({Slice(), 0, Slice(), 1}) - _f.index({Slice(), 0, Slice(), 3})) +
            1.0 / 6.0 * _value * velocity);
  }
}

template <>
void
LBMFixedZerothOrderBCTempl<19>::bottomBoundary()
{
  // TBD
}

template <>
void
LBMFixedZerothOrderBCTempl<27>::bottomBoundary()
{
  // TBD
}

template <>
void
LBMFixedZerothOrderBCTempl<9>::topBoundary()
{
  torch::Tensor velocity = (_f.index({Slice(), _grid_size[1] - 1, Slice(), 0}) +
                            _f.index({Slice(), _grid_size[1] - 1, Slice(), 1}) +
                            _f.index({Slice(), _grid_size[1] - 1, Slice(), 3}) +
                            2 * (_f.index({Slice(), _grid_size[1] - 1, Slice(), 2}) +
                                 _f.index({Slice(), _grid_size[1] - 1, Slice(), 5}) +
                                 _f.index({Slice(), _grid_size[1] - 1, Slice(), 6}))) /
                               _value -
                           1.0;

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._bottom[0]];
  _u.index_put_({Slice(), _grid_size[1] - 1, Slice(), opposite_dir},
                _f.index({Slice(), _grid_size[1] - 1, Slice(), _stencil._bottom[0]}) -
                    2.0 / 3.0 * _value * velocity);

  // other directions
  for (unsigned int i = 1; i < _stencil._bottom.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._bottom[i]];
    _u.index_put_({Slice(), _grid_size[1] - 1, Slice(), opposite_dir},
                  _f.index({Slice(), _grid_size[1] - 1, Slice(), _stencil._bottom[i]}) +
                      0.5 * _stencil._ex[opposite_dir] *
                          (_f.index({Slice(), _grid_size[1] - 1, Slice(), 3}) -
                           _f.index({Slice(), _grid_size[1] - 1, Slice(), 1})) -
                      1.0 / 6.0 * _value * velocity);
  }
}

template <>
void
LBMFixedZerothOrderBCTempl<19>::topBoundary()
{
  // TBD
}

template <>
void
LBMFixedZerothOrderBCTempl<27>::topBoundary()
{
  // TBD
}

template <int dimension>
void
LBMFixedZerothOrderBCTempl<dimension>::computeBuffer()
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

template class LBMFixedZerothOrderBCTempl<9>;
template class LBMFixedZerothOrderBCTempl<19>;
template class LBMFixedZerothOrderBCTempl<27>;
