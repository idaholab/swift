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

  _specular_reflection_indices = torch::zeros({_boundary_indices.size(0)}, MooseTensor::intTensorOptions());
  _specular_reflection_indices.fill_(0);

  // determineBoundaryTypes();
  int64_t row_index = 0;
  int64_t k = 0;
  for (int64_t i = 0; i < expected_shape[0]; i++)
    for (int64_t j = 0; j < expected_shape[1]; j++)
      for (int64_t ic = 0; ic < expected_shape[3]; ic++)
      {
        int64_t boundary_type = _boundary_types[i][j][k].item<int64_t>();
        if (boundary_type != -1)
        {
          int64_t if_stream_index = _all_boundary_types.size(0) * ic + boundary_type;

          // when streaming along ic is NOT possible at i, j, k i.e zero
          if (_if_stream[if_stream_index].item<int64_t>() == 0)
          {
            /*
            _if_stream knows which directions are allowed to stream based on the boundary type
              and _boundary_mask[i][j][k][ic] = 1; means streaming at that location with that direction is not allowed
              so boundary condition will be applied
            */
            // _boundary_indices[i][j][k][ic] = 1;

            // _icsr was precomputed using MATLAB so indices start from 1
            _specular_reflection_indices[row_index] = _icsr[if_stream_index] - 1;
            row_index++;
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
    buildBoundaryIndices();
  }

  // bounce-back
  _u.index_put_({_boundary_indices.index({Slice(), 0}), 
                _boundary_indices.index({Slice(), 1}), 
                _boundary_indices.index({Slice(), 2}),
                _stencil._op.index_select(0, _boundary_indices.index({Slice(), 3}))}, 
      _r * _f_old[0].index({_boundary_indices.index({Slice(), 0}), 
              _boundary_indices.index({Slice(), 1}), 
              _boundary_indices.index({Slice(), 2}), 
              _boundary_indices.index({Slice(), 3})}));
  
  // specular reflection
  _u.index_put_({_boundary_indices.index({Slice(), 0}), 
              _boundary_indices.index({Slice(), 1}), 
              _boundary_indices.index({Slice(), 2}),
              _specular_reflection_indices}, 
    (1 -_r) * _f_old[0].index({_boundary_indices.index({Slice(), 0}), 
            _boundary_indices.index({Slice(), 1}), 
            _boundary_indices.index({Slice(), 2}), 
            _boundary_indices.index({Slice(), 3})}));
}

void
LBMSpecularReflectionBoundary::computeBuffer()
{
  const auto n_old = _f_old.size();
  if (n_old != 0)
    wallBoundary();
  _lb_problem.maskedFillSolids(_u, 0);
}
