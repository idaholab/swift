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

registerMooseObject("SwiftApp", LBMFixedFirstOrderBC9Q);
registerMooseObject("SwiftApp", LBMFixedFirstOrderBC19Q);
registerMooseObject("SwiftApp", LBMFixedFirstOrderBC27Q);

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
LBMFixedFirstOrderBCTempl<9>::frontBoundary()
{
  // There is no front boundary in 2D
  mooseError("There is no front boundary in 2 dimensions.");
}

template <>
void
LBMFixedFirstOrderBCTempl<19>::frontBoundary()
{
  // TBD
}

template <>
void
LBMFixedFirstOrderBCTempl<27>::frontBoundary()
{
  // TBD
}

template <>
void
LBMFixedFirstOrderBCTempl<9>::backBoundary()
{
  // There is no back boundary in 2D
  mooseError("There is no back boundary in 2 dimensions.");
}

template <>
void
LBMFixedFirstOrderBCTempl<19>::backBoundary()
{
  // TBD
}

template <>
void
LBMFixedFirstOrderBCTempl<27>::backBoundary()
{
  // TBD
}

template <>
void
LBMFixedFirstOrderBCTempl<9>::leftBoundary()
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
LBMFixedFirstOrderBCTempl<19>::leftBoundary()
{
  // TBD
}

template <>
void
LBMFixedFirstOrderBCTempl<27>::leftBoundary()
{
  torch::Tensor density = 1.0 / (1.0 - _value) *
                          (torch::sum(_f.index({0, Slice(), Slice(), -_stencil._neutral_x}), -1) +
                           2 * torch::sum(_f.index({0, Slice(), Slice(), _stencil._right}), -1));

  _u.index_put_({0, Slice(), Slice(), _stencil._left[0]},
                _f.index({0, Slice(), Slice(), _stencil._right[0]}) +
                    2.0 * _stencil._weights[_stencil._left[0]] / _lb_problem._cs2 * _value *
                        density);

  for (unsigned int i = 1; i < _stencil._left.size(0); i++)
  {
    auto n1 = torch::sum(_f.index({0, Slice(), Slice(), _stencil._neutral_x_pos_y}), -1);
    auto n2 = torch::sum(_f.index({0, Slice(), Slice(), _stencil._neutral_x_neg_y}), -1);
    auto n3 = torch::sum(_f.index({0, Slice(), Slice(), _stencil._neutral_x_pos_z}), -1);
    auto n4 = torch::sum(_f.index({0, Slice(), Slice(), _stencil._neutral_x_neg_z}), -1);

    _u.index_put_({0, Slice(), Slice(), _stencil._left[i]},
                  _f.index({0, Slice(), Slice(), _stencil._right[i]}) +
                      2.0 * _stencil._weights[_stencil._left[i]] / _lb_problem._cs2 * _value *
                          density -
                      0.5 * _stencil._ey[_stencil._left[i]] * (n1 - n2) -
                      0.5 * _stencil._ez[_stencil._left[i]] * (n3 - n4));
  }
}

template <>
void
LBMFixedFirstOrderBCTempl<9>::rightBoundary()
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
LBMFixedFirstOrderBCTempl<19>::rightBoundary()
{
  // TBD
}

template <>
void
LBMFixedFirstOrderBCTempl<27>::rightBoundary()
{
  torch::Tensor density = 1.0 / (1.0 + _value) *
                          (torch::sum(_f.index({0, Slice(), Slice(), -_stencil._neutral_x}), -1) +
                           2 * torch::sum(_f.index({0, Slice(), Slice(), _stencil._left}), -1));

  _u.index_put_({0, Slice(), Slice(), _stencil._right[0]},
                _f.index({0, Slice(), Slice(), _stencil._left[0]}) -
                    2.0 * _stencil._weights[_stencil._right[0]] / _lb_problem._cs2 * _value *
                        density);

  for (unsigned int i = 1; i < _stencil._right.size(0); i++)
  {
    auto n1 =
        torch::sum(_f.index({0, Slice(), Slice(), _stencil._neutral_x_pos_y}), -1).unsqueeze(-1);
    auto n2 =
        torch::sum(_f.index({0, Slice(), Slice(), _stencil._neutral_x_neg_y}), -1).unsqueeze(-1);

    auto n3 =
        torch::sum(_f.index({0, Slice(), Slice(), _stencil._neutral_x_pos_z}), -1).unsqueeze(-1);
    auto n4 =
        torch::sum(_f.index({0, Slice(), Slice(), _stencil._neutral_x_neg_z}), -1).unsqueeze(-1);

    _u.index_put_({0, Slice(), Slice(), _stencil._right[i]},
                  _f.index({0, Slice(), Slice(), _stencil._left[i]}) -
                      2.0 * _stencil._weights[_stencil._right[i]] / _lb_problem._cs2 * _value *
                          density +
                      0.5 * _stencil._ey[_stencil._right[i]] * (n1 - n2) +
                      0.5 * _stencil._ez[_stencil._right[i]] * (n3 - n4));
  }
}

template <>
void
LBMFixedFirstOrderBCTempl<9>::bottomBoundary()
{
  torch::Tensor density =
      1.0 / (1.0 - _value) *
      (_f.index({Slice(), 0, Slice(), 0}) + _f.index({Slice(), 0, Slice(), 1}) +
       _f.index({Slice(), 0, Slice(), 3}) +
       2 * (_f.index({Slice(), 0, Slice(), 4}) + _f.index({Slice(), 0, Slice(), 7}) +
            _f.index({Slice(), 0, Slice(), 8})));

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._bottom[0]];
  _u.index_put_({Slice(), 0, Slice(), _stencil._bottom[0]},
                _f.index({Slice(), 0, Slice(), opposite_dir}) + 2.0 / 3.0 * density * _value);

  // other directions
  for (unsigned int i = 1; i < _stencil._bottom.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._bottom[i]];
    _u.index_put_(
        {Slice(), 0, Slice(), _stencil._bottom[i]},
        _f.index({Slice(), 0, Slice(), opposite_dir}) -
            0.5 * _stencil._ex[_stencil._bottom[i]] *
                (_f.index({Slice(), 0, Slice(), 1}) - _f.index({Slice(), 0, Slice(), 3})) +
            1.0 / 6.0 * density * _value);
  }
}

template <>
void
LBMFixedFirstOrderBCTempl<19>::bottomBoundary()
{
  // TBD
}

template <>
void
LBMFixedFirstOrderBCTempl<27>::bottomBoundary()
{
  // TBD
}

template <>
void
LBMFixedFirstOrderBCTempl<9>::topBoundary()
{
  torch::Tensor density = 1.0 / (1.0 + _value) *
                          (_f.index({Slice(), _grid_size[1] - 1, Slice(), 0}) +
                           _f.index({Slice(), _grid_size[1] - 1, Slice(), 1}) +
                           _f.index({Slice(), _grid_size[1] - 1, Slice(), 3}) +
                           2 * (_f.index({Slice(), _grid_size[1] - 1, Slice(), 2}) +
                                _f.index({Slice(), _grid_size[1] - 1, Slice(), 5}) +
                                _f.index({Slice(), _grid_size[1] - 1, Slice(), 6})));

  // axis aligned direction
  const auto & opposite_dir = _stencil._op[_stencil._bottom[0]];
  _u.index_put_({Slice(), _grid_size[1] - 1, Slice(), opposite_dir},
                _f.index({Slice(), _grid_size[1] - 1, Slice(), _stencil._bottom[0]}) -
                    2.0 / 3.0 * density * _value);

  // other directions
  for (unsigned int i = 1; i < _stencil._bottom.size(0); i++)
  {
    const auto & opposite_dir = _stencil._op[_stencil._bottom[i]];
    _u.index_put_({Slice(), _grid_size[1] - 1, Slice(), opposite_dir},
                  _f.index({Slice(), _grid_size[1] - 1, Slice(), _stencil._bottom[i]}) +
                      0.5 * _stencil._ex[opposite_dir] *
                          (_f.index({Slice(), _grid_size[1] - 1, Slice(), 3}) -
                           _f.index({Slice(), _grid_size[1] - 1, Slice(), 1})) -
                      1.0 / 6.0 * density * _value);
  }
}

template <>
void
LBMFixedFirstOrderBCTempl<19>::topBoundary()
{
  // TBD
}

template <>
void
LBMFixedFirstOrderBCTempl<27>::topBoundary()
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

template class LBMFixedFirstOrderBCTempl<9>;
template class LBMFixedFirstOrderBCTempl<19>;
template class LBMFixedFirstOrderBCTempl<27>;
