/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMCollisionDynamics.h"

registerMooseObject("SwiftApp", LBMBGKCollision);
registerMooseObject("SwiftApp", LBMMRTCollision);

template<int coll_dyn>
InputParameters
LBMCollisionDynamicsTempl<coll_dyn>::validParams()
{
  InputParameters params = LatticeBoltzmannOperator::validParams();

  params.addClassDescription("Template object for LBM collision dynamics");
  params.addRequiredParam<TensorInputBufferName>("f", "Input buffer distribution function");
  params.addRequiredParam<TensorInputBufferName>("feq", "Input buffer equilibrium distribution function");
  
  params.addParam<Real>("tau_0", 1.0, "Relaxation parameter");
  params.addParam<bool>("projection", false, "Whether or not to project non-equilibrium onto Hermite space.");

  return params;
}

template<int coll_dyn>
LBMCollisionDynamicsTempl<coll_dyn>::LBMCollisionDynamicsTempl(const InputParameters & parameters)
  : LatticeBoltzmannOperator(parameters),
  _f(getInputBuffer("f")),
  _feq(getInputBuffer("feq")),
  _shape(_lb_problem.getGridSize()),
  _tau_0(getParam<Real>("tau_0")),
  _projection(getParam<bool>("projection"))
{
}

template<>
void
LBMCollisionDynamicsTempl<0>::BGKDynamics()
{ 
  // LBM BGK collision
  _u = _f - 1.0 / _tau_0 * (_f - _feq);
  _lb_problem.maskedFillSolids(_u, 0);
}

template<>
void
LBMCollisionDynamicsTempl<0>::MRTDynamics()
{ 
  // LBM BGK collision
}

template<int coll_dyn>
void
LBMCollisionDynamicsTempl<coll_dyn>::computeBuffer()
{
  switch (coll_dyn)
  {
    case 0:
      BGKDynamics();
      break;
    case 1:
      MRTDynamics();
      break;
    default:
      mooseError("Undefined template value");
  }
}

template class LBMCollisionDynamicsTempl<0>;
template class LBMCollisionDynamicsTempl<1>;
