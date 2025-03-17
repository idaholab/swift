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
        "f_old", "Buffer with the reciprocal of the integrated buffer");
    return params;
}


LBMSpecularReflectionBoundary::LBMSpecularReflectionBoundary(const InputParameters & parameters)
    : LBMBoundaryCondition(parameters),
    _f_old(_lb_problem.getBufferOld(getParam<TensorInputBufferName>("f_old"), 1)),
    _r(getParam<Real>("r"))
{
}


void
LBMSpecularReflectionBoundary::buildBoundaryMask()
{
  /**
   * Building boundary mask
   */
  std::vector<int64_t> expected_shape = {_mesh.getElementsInDimension(0),
                                        _mesh.getElementsInDimension(1),
                                        _mesh.getElementsInDimension(2),
                                        _stencil._q};

  const torch::Tensor & mesh_expanded = _mesh.getBinaryMesh().unsqueeze(3).expand(expected_shape);
  _boundary_mask = (mesh_expanded == 2) & (_u == 0);
  _boundary_mask = _boundary_mask.to(torch::kBool);

  _specular_reflection_indices = torch::zeros(expected_shape, MooseTensor::intTensorOptions());
  _specular_reflection_indices.fill_(0);

  determineBoundaryTypes();

  int k = 0;
  for (int i = 0; i < expected_shape[0]; i++)
    for (int j = 0; j < expected_shape[1]; j++)
      for (int ic = 0; ic < _stencil._q; ic++)
      {
        if (_boundary_types[i][j][k].item<int>()== -1)
          _specular_reflection_indices[i][j][k][ic] = ic;

        else
        {
          int if_stream_index = _all_boundary_types.size(0) * ic + _boundary_types[i][j][k].item<int>();

          // when streaming along ic is possible at i, j, k
          if (_if_stream[if_stream_index].item<int>() != 0)
            _specular_reflection_indices[i][j][k][ic] = ic;

          // when streaming along ic is NOT possible at i, j, k
          else
          {
            /*
            _if_stream knows which directions are allowed to stream based on the boundary type
              and _boundary_mask[i][j][k][ic] = 1; means streaming at that location with that direction is not allowed
              so boundary condition will be applied
            */
            _boundary_mask[i][j][k][ic] = 1;

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
  std::vector<int> d2q9_order_to_new_order{7, 3, 6, 4, 0, 2, 8, 1, 5};

  int k = 0;

  for (int i = 0; i < _mesh.getElementsInDimension(0); i++)
    for (int j = 0; j < _mesh.getElementsInDimension(1); j++)
    {
      if (binary_mesh[i][j][k].item<int>() != 0)
      {
        // assembling binary string
        std::string string_of_binary_digits;
        // search in all directions
        for (int ic = 0; ic < _stencil._q; ic++)
        {
          int i_prime = i + _stencil._ex[d2q9_order_to_new_order[ic]].item<int>();
          int j_prime = j + _stencil._ey[d2q9_order_to_new_order[ic]].item<int>();

          // ensuring periodicity
          i_prime = (i_prime < 0) ? i_prime + _mesh.getElementsInDimension(0) : i_prime;
          i_prime = (i_prime >= _mesh.getElementsInDimension(0)) ? i_prime - _mesh.getElementsInDimension(0) : i_prime;

          j_prime = (j_prime < 0) ? j_prime + _mesh.getElementsInDimension(1) : j_prime;
          j_prime = (j_prime >= _mesh.getElementsInDimension(1)) ? j_prime - _mesh.getElementsInDimension(1) : j_prime;

          //
          if (binary_mesh[i_prime][j_prime][k].item<int>() == 0)
            string_of_binary_digits += '0';
          else
            string_of_binary_digits += '1';
        }
        // convert binary to decimal
        std::bitset<32> binary(string_of_binary_digits);
        int decimal_number = binary.to_ulong();
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
            auto index = torch::nonzero(comparison_result).item<int>();

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
    buildBoundaryMask();
  }

  torch::Tensor f_bounce_back = torch::zeros_like(_f_old[0]);
  torch::Tensor f_specular_reflect = torch::zeros_like(_f_old[0]);

  // Bounce-back
  for (int ic = 0; ic < _stencil._q; ic++)
  {
    int bounce_back_index = _stencil._op[ic].item<int64_t>();
    auto lattice_slice = _f_old[0].index({Slice(), Slice(), Slice(), ic});
    f_bounce_back.index_put_({Slice(), Slice(), Slice(), bounce_back_index}, lattice_slice);
  }

  // Specular reflection
  f_specular_reflect.index_put_({Slice(), Slice(), Slice(), Slice()},
                   _f_old[0].gather(3, _specular_reflection_indices.to(torch::kInt64)));

  // Combine bounce-back and specular reflection
  torch::Tensor combined_boundary_conditions = _r * f_bounce_back + (1 - _r) * f_specular_reflect;
  _u.index_put_({_boundary_mask}, combined_boundary_conditions.index({_boundary_mask}));
}

void
LBMSpecularReflectionBoundary::computeBuffer()
{
  const auto n_old = _f_old.size();
  if (n_old != 0)
    wallBoundary();
  _lb_problem.maskedFillSolids(_u, 0);
}
