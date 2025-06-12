/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "SwiftConstantInterface.h"
#include "InputParameters.h"

SwiftConstantInterface::SwiftConstantInterface(const InputParameters & params)
  : _params(params),
    _sci_tensor_problem(*params.getCheckedPointerParam<TensorProblem *>("_tensor_problem"))
{
}
