/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorTimeIntegrator.h"
#include "TensorProblem.h"

InputParameters
TensorTimeIntegrator::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.registerBase("TensorTimeIntegrator");
  params.addClassDescription("TensorTimeIntegrator object.");
  return params;
}

TensorTimeIntegrator::TensorTimeIntegrator(const InputParameters & parameters)
  : TensorOperator(parameters), _sub_dt(_tensor_problem.subDt())
{
}

const std::vector<torch::Tensor> &
TensorTimeIntegrator::getBufferOld(const std::string & param, unsigned int max_states)
{
  return getBufferOldByName(getParam<TensorInputBufferName>(param), max_states);
}

const std::vector<torch::Tensor> &
TensorTimeIntegrator::getBufferOldByName(const TensorInputBufferName & buffer_name,
                                      unsigned int max_states)
{
  return _tensor_problem.getBufferOld(buffer_name, max_states);
}
