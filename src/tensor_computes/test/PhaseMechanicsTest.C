/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "PhaseMechanicsTest.h"
#include "SwiftUtils.h"
#include "TensorProblem.h"
#include <ATen/ops/zeros.h>

registerMooseObject("SwiftApp", PhaseMechanicsTest);

InputParameters
PhaseMechanicsTest::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("DeGeus mechanics test material.");
  return params;
}

PhaseMechanicsTest::PhaseMechanicsTest(const InputParameters & parameters)
  : TensorOperator<>(parameters)
{
}

void
PhaseMechanicsTest::computeBuffer()
{
  _u = torch::zeros(_domain.getShape(), MooseTensor::floatTensorOptions());

  // Define the slicing indices
  int64_t s = _dim == 2 ? 30 : 9;
  auto slice1 = torch::indexing::Slice(-s, torch::indexing::None);
  auto slice2 = torch::indexing::Slice(torch::indexing::None, s);
  auto slice3 = torch::indexing::Slice(-s, torch::indexing::None);

  // Perform the slicing operation equivalent to phase[-9:,:9,-9:] = 1.0 in Python
  if (_dim == 3)
    _u.index_put_({slice1, slice2, slice3}, 1.0);
  else if (_dim == 2)
    _u.index_put_({slice1, slice2}, 1.0);
  else
    mooseError("Unsupported problem dimension");
}
