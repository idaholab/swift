/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMComputeForces.h"
#include "LatticeBoltzmannProblem.h"

using namespace torch::indexing;

registerMooseObject("SwiftApp", LBMComputeForces);

InputParameters
LBMComputeForces::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();
  // params.addRequiredParam<TensorInputBufferName>("f", "Distribution function");
  params.addRequiredParam<TensorInputBufferName>("temperature", "Macroscopic temperature");
  params.addRequiredParam<std::string>("rho0", "Macroscopic density");
  params.addRequiredParam<std::string>("T0", "Reference temperature");
  params.addParam<bool>("enable_gravity", false, "Whether to consider gravity or not");
  params.addParam<std::string>("gravity", "1.0", "Gravitational accelaration");
  params.addClassDescription("Compute object for LB forces");
  return params;
}

LBMComputeForces::LBMComputeForces(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters), /*_f(getInputBuffer("f"))*/
    _temperature(getInputBufferByName(getParam<TensorInputBufferName>("temperature"))),
    _density(_lb_problem.getConstant<Real>(getParam<std::string>("rho0"))),
    _T0(_lb_problem.getConstant<Real>(getParam<std::string>("T0"))),
    _enable_gravity(getParam<bool>("enable_gravity")),
    _g(_lb_problem.getConstant<Real>(getParam<std::string>("gravity")))
{
}

void
LBMComputeForces::computeBuffer()
{
  if (_enable_gravity)
    _u.index_put_({Slice(), Slice(), Slice(), 1}, (-1.0 * _g * _density) * (_temperature - _T0));
  else
  {
    // TBD
  }
  _lb_problem.maskedFillSolids(_u, 0);
}
