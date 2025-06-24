/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMFixedFirstOrderBC.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannMesh.h"
#include "LatticeBoltzmannStencilBase.h"

#include <cstdlib>

using namespace torch::indexing;

registerMooseObject("SwiftApp", LBMFixedFirstOrderBC2D);
registerMooseObject("SwiftApp", LBMFixedFirstOrderBC3D);

template <int dimension>
InputParameters
LBMFixedFirstOrderBCTempl<dimension>::validParams()
{
  InputParameters params = LBMBoundaryCondition::validParams();
  params.addClassDescription("LBMFixedFirstOrderBC object");
  params.addRequiredParam<TensorInputBufferName>("f", "Input buffer distribution function");
  params.addRequiredParam<std::string>("value", "Fixed input velocity");
  params.addParam<bool>("perturb", false, "Whether to perturb first order moment at the boundary");
  return params;
}

template <int dimension>
LBMFixedFirstOrderBCTempl<dimension>::LBMFixedFirstOrderBCTempl(const InputParameters & parameters)
  : LBMBoundaryCondition(parameters),
    _f(getInputBufferByName(getParam<TensorInputBufferName>("f"))),
    _grid_size(_lb_problem.getGridSize()),
    _value(_lb_problem.getConstant<Real>(getParam<std::string>("value"))),
    _perturb(getParam<bool>("perturb"))
{
}

template <>
void
LBMFixedFirstOrderBCTempl<2>::topBoundary()
{
  // There is no top boundary in 2D
}

template <>
void
LBMFixedFirstOrderBCTempl<3>::topBoundary()
{
  // TBD
}

template <>
void
LBMFixedFirstOrderBCTempl<2>::bottomBoundary()
{
  // There is no bottom boundary in 2D
}

template <>
void
LBMFixedFirstOrderBCTempl<3>::bottomBoundary()
{
  // TBD
}

template <>
void
LBMFixedFirstOrderBCTempl<2>::leftBoundary()
{
  Real deltaU = 0.0;
  torch::Tensor u_x_perturbed = torch::zeros({_grid_size[1], 1}, MooseTensor::floatTensorOptions());

  if (_perturb)
  {
    deltaU = 1.0e-4 * _value;
    Real phi = 0.0; // static_cast<Real>(rand()) / static_cast<float>(RAND_MAX) * 2.0 * M_PI;
    torch::Tensor y_coords =
        torch::arange(0, _grid_size[1], MooseTensor::floatTensorOptions()).unsqueeze(1);
    u_x_perturbed = _value + deltaU * torch::sin(y_coords / _grid_size[1] * 2.0 * M_PI + phi);
  }
  else
    u_x_perturbed.fill_(_value);

  torch::Tensor density =
      1.0 / (1.0 - u_x_perturbed) *
      (_f.index({0, Slice(), Slice(), 0}) + _f.index({0, Slice(), Slice(), 2}) +
       _f.index({0, Slice(), Slice(), 4}) +
       2.0 * (_f.index({0, Slice(), Slice(), 3}) + _f.index({0, Slice(), Slice(), 6}) +
              _f.index({0, Slice(), Slice(), 7})));

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._left[0]];
  _u.index_put_({0, Slice(), Slice(), _stencil._left[0]},
                _f.index({0, Slice(), Slice(), opposite_dir}) +
                    2.0 / 3.0 * density * u_x_perturbed);

  // other directions
  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._left[i]];
    _u.index_put_(
        {0, Slice(), Slice(), _stencil._left[i]},
        _f.index({0, Slice(), Slice(), opposite_dir}) -
            0.5 * _stencil._ey[_stencil._left[i]] *
                (_f.index({0, Slice(), Slice(), 2}) - _f.index({0, Slice(), Slice(), 4})) +
            1.0 / 6.0 * density * u_x_perturbed);
  }
}

template <>
void
LBMFixedFirstOrderBCTempl<3>::leftBoundary()
{
  // TBD
}

template <>
void
LBMFixedFirstOrderBCTempl<2>::rightBoundary()
{
  torch::Tensor density = 1.0 / (1.0 + _value) *
                          (_f.index({_grid_size[0] - 1, Slice(), Slice(), 0}) +
                           _f.index({_grid_size[0] - 1, Slice(), Slice(), 2}) +
                           _f.index({_grid_size[0] - 1, Slice(), Slice(), 4}) +
                           2 * (_f.index({_grid_size[0] - 1, Slice(), Slice(), 1}) +
                                _f.index({_grid_size[0] - 1, Slice(), Slice(), 5}) +
                                _f.index({_grid_size[0] - 1, Slice(), Slice(), 8})));

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._left[0]];
  _u.index_put_({_grid_size[0] - 1, Slice(), Slice(), opposite_dir},
                _f.index({_grid_size[0] - 1, Slice(), Slice(), _stencil._left[0]}) -
                    2.0 / 3.0 * density * _value);

  // other directions
  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._left[i]];
    _u.index_put_({_grid_size[0] - 1, Slice(), Slice(), opposite_dir},
                  _f.index({_grid_size[0] - 1, Slice(), Slice(), _stencil._left[i]}) +
                      0.5 * _stencil._ey[opposite_dir] *
                          (_f.index({_grid_size[0] - 1, Slice(), Slice(), 4}) -
                           _f.index({_grid_size[0] - 1, Slice(), Slice(), 2})) -
                      1.0 / 6.0 * density * _value);
  }
}

template <>
void
LBMFixedFirstOrderBCTempl<3>::rightBoundary()
{
  // TBD
}

template <>
void
LBMFixedFirstOrderBCTempl<2>::frontBoundary()
{
  torch::Tensor density =
      1.0 / (1.0 - _value) *
      (_f.index({Slice(), 0, Slice(), 0}) + _f.index({Slice(), 0, Slice(), 1}) +
       _f.index({Slice(), 0, Slice(), 3}) +
       2 * (_f.index({Slice(), 0, Slice(), 4}) + _f.index({Slice(), 0, Slice(), 7}) +
            _f.index({Slice(), 0, Slice(), 8}))) /
      _value;

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._front[0]];
  _u.index_put_({Slice(), 0, Slice(), _stencil._front[0]},
                _f.index({Slice(), 0, Slice(), opposite_dir}) + 2.0 / 3.0 * density * _value);

  // other directions
  for (unsigned int i = 1; i < _stencil._front.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._front[i]];
    _u.index_put_(
        {Slice(), 0, Slice(), _stencil._front[i]},
        _f.index({Slice(), 0, Slice(), opposite_dir}) -
            0.5 * _stencil._ex[_stencil._front[i]] *
                (_f.index({Slice(), 0, Slice(), 1}) - _f.index({Slice(), 0, Slice(), 3})) +
            1.0 / 6.0 * density * _value);
  }
}

template <>
void
LBMFixedFirstOrderBCTempl<3>::frontBoundary()
{
  // TBD
}

template <>
void
LBMFixedFirstOrderBCTempl<2>::backBoundary()
{
  torch::Tensor density = 1.0 / (1.0 + _value) *
                          (_f.index({Slice(), _grid_size[1] - 1, Slice(), 0}) +
                           _f.index({Slice(), _grid_size[1] - 1, Slice(), 1}) +
                           _f.index({Slice(), _grid_size[1] - 1, Slice(), 3}) +
                           2 * (_f.index({Slice(), _grid_size[1] - 1, Slice(), 2}) +
                                _f.index({Slice(), _grid_size[1] - 1, Slice(), 5}) +
                                _f.index({Slice(), _grid_size[1] - 1, Slice(), 6})));

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._front[0]];
  _u.index_put_({Slice(), _grid_size[1] - 1, Slice(), opposite_dir},
                _f.index({Slice(), _grid_size[1] - 1, Slice(), _stencil._front[0]}) -
                    2.0 / 3.0 * density * _value);

  // other directions
  for (unsigned int i = 1; i < _stencil._front.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._front[i]];
    _u.index_put_({Slice(), _grid_size[1] - 1, Slice(), opposite_dir},
                  _f.index({Slice(), _grid_size[1] - 1, Slice(), _stencil._front[i]}) +
                      0.5 * _stencil._ey[opposite_dir] *
                          (_f.index({Slice(), _grid_size[1] - 1, Slice(), 4}) -
                           _f.index({Slice(), _grid_size[1] - 1, Slice(), 2})) -
                      1.0 / 6.0 * density * _value);
  }
}

template <>
void
LBMFixedFirstOrderBCTempl<3>::backBoundary()
{
  // TBD
}

template <int dimension>
void
LBMFixedFirstOrderBCTempl<dimension>::computeBuffer()
{
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

template class LBMFixedFirstOrderBCTempl<2>;
template class LBMFixedFirstOrderBCTempl<3>;
