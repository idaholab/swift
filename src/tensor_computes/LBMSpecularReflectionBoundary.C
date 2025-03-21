/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMSpecularReflectionBoundary.h"
#include "LBMBoundaryCondition.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannMesh.h"

using namespace torch::indexing;

registerMooseObject("SwiftApp", LBMSpecularReflectionBoundary);

InputParameters
LBMSpecularReflectionBoundary::validParams()
{
    InputParameters params = LBMBoundaryCondition::validParams();
    params.addParam<Real>("r", 0.5, "Combination coefficeint");
    params.addClassDescription("LBM combination of bounce-back and specular reflection boundary condition");
    params.addRequiredParam<TensorInputBufferName>(
        "f_old", "Old timestep distribution function");
    return params;
}

LBMSpecularReflectionBoundary::LBMSpecularReflectionBoundary(const InputParameters & parameters)
    : LBMBoundaryCondition(parameters),
    _f_old(_lb_problem.getBufferOld(getParam<TensorInputBufferName>("f_old"), 1)),
    _r(getParam<Real>("r"))
{
}

void
LBMSpecularReflectionBoundary::buildBoundaryIndices()
{
  /**
   * Building boundary mask
   */
  std::vector<int64_t> expected_shape = {_mesh.getElementsInDimension(0),
                                        _mesh.getElementsInDimension(1),
                                        _mesh.getElementsInDimension(2),
                                        _stencil._q};

  // call parent class method
  LBMBoundaryCondition::buildBoundaryIndices();

  _specular_reflection_indices = torch::zeros(expected_shape, MooseTensor::intTensorOptions());
  _specular_reflection_indices.fill_(0);

  determineBoundaryTypes();

  int64_t k = 0;
  for (int64_t i = 0; i < expected_shape[0]; i++)
    for (int64_t j = 0; j < expected_shape[1]; j++)
      for (int64_t ic = 0; ic < _stencil._q; ic++)
      {
        if (_boundary_types[i][j][k].item<int64_t>()== -1)
          _specular_reflection_indices[i][j][k][ic] = ic;

        else
        {
          int64_t if_stream_index = _all_boundary_types.size(0) * ic + _boundary_types[i][j][k].item<int64_t>();

          // when streaming along ic is possible at i, j, k, i.e nonzero
          if (_if_stream[if_stream_index].item<int64_t>() != 0)
            _specular_reflection_indices[i][j][k][ic] = ic;

          // when streaming along ic is NOT possible at i, j, k i.e zero
          else
          {
            /*
            _if_stream knows which directions are allowed to stream based on the boundary type
              and _boundary_mask[i][j][k][ic] = 1; means streaming at that location with that direction is not allowed
              so boundary condition will be applied
            */
            // _boundary_indices[i][j][k][ic] = 1;

            // _icsr was precomputed using MATLAB so indices start from 1
            _specular_reflection_indices[i][j][k][ic] = _icsr[if_stream_index] - 1;
          }
        }
      }
}

void
LBMSpecularReflectionBoundary::determineBoundaryTypes()
{
  /**
   * Scan the binary domain in a 3x3 window to determine boundary types
   * D2Q9 only
   */

  const torch::Tensor & binary_mesh =  _mesh.getBinaryMesh();
  _boundary_types = torch::empty(_lb_problem.getShape(), MooseTensor::intTensorOptions());

  // (-1) indicates boundary hit
  _boundary_types.fill_(-1);

  // changes the order of D2Q9
  std::vector<int64_t> d2q9_order_to_new_order{7, 3, 6, 4, 0, 2, 8, 1, 5};

  int64_t k = 0;

  for (int64_t i = 0; i < _mesh.getElementsInDimension(0); i++)
    for (int64_t j = 0; j < _mesh.getElementsInDimension(1); j++)
    {
      if (binary_mesh[i][j][k].item<int64_t>() != 0)
      {
        // assembling binary string
        std::string string_of_binary_digits;
        // search in all directions
        for (int64_t ic = 0; ic < _stencil._q; ic++)
        {
          int64_t i_prime = i + _stencil._ex[d2q9_order_to_new_order[ic]].item<int64_t>();
          int64_t j_prime = j + _stencil._ey[d2q9_order_to_new_order[ic]].item<int64_t>();

          // ensuring periodicity
          i_prime = (i_prime < 0) ? i_prime + _mesh.getElementsInDimension(0) : i_prime;
          i_prime = (i_prime >= _mesh.getElementsInDimension(0)) ? i_prime - _mesh.getElementsInDimension(0) : i_prime;

          j_prime = (j_prime < 0) ? j_prime + _mesh.getElementsInDimension(1) : j_prime;
          j_prime = (j_prime >= _mesh.getElementsInDimension(1)) ? j_prime - _mesh.getElementsInDimension(1) : j_prime;

          //
          if (binary_mesh[i_prime][j_prime][k].item<int64_t>() == 0)
            string_of_binary_digits += '0';
          else
            string_of_binary_digits += '1';
        }

        // convert binary to decimal
        std::bitset<32> binary(string_of_binary_digits);
        int64_t decimal_number = binary.to_ulong();
        if (decimal_number != 511)
        {
          auto comparison_result = (_all_boundary_types == decimal_number);
          bool is_decimal_number_in_boundary_types = torch::any(comparison_result).item<bool>();

          if (!is_decimal_number_in_boundary_types)
          {
            std::string error_message = "Boundary type " + std::to_string(decimal_number) + " is not found";
            mooseError(error_message);
          }
          else
          {
            // the index where boundary type matches decimal_number
            int64_t index = torch::nonzero(comparison_result).item<int64_t>();

            _boundary_types[i][j][k] = index;
          }
        }
      }
    }
}

void
LBMSpecularReflectionBoundary::wallBoundary()
{
  // build boundary mask in the begining of simulation
  if (_lb_problem.getTotalSteps() == 0)
  {
    _f_specular_reflect = torch::zeros_like(_f_old[0]);
    buildBoundaryIndices();
  }

  // Specular reflection
  _f_specular_reflect.index_put_({Slice(), Slice(), Slice(), Slice()},
                   _f_old[0].gather(3, _specular_reflection_indices));

  for (int ic = 1; ic < _stencil._q; ic++)
  { 
    const auto & opposite_dir = _stencil._op[_stencil._front[ic]];
    _u.index_put_({_boundary_indices[Slice(), 0], _boundary_indices[Slice(), 1], _boundary_indices[Slice(), 2], ic}, 
                  _r * _f_old[0].index({_boundary_indices[Slice(), 0], _boundary_indices[Slice(), 1], _boundary_indices[Slice(), 2], opposite_dir}) + \
                  (1 - _r) * _f_specular_reflect.index({_boundary_indices[Slice(), 0], _boundary_indices[Slice(), 1], _boundary_indices[Slice(), 2], ic}));
  }
}

void
LBMSpecularReflectionBoundary::computeBuffer()
{
  const auto n_old = _f_old.size();
  if (n_old != 0)
    wallBoundary();
  _lb_problem.maskedFillSolids(_u, 0);
}
