/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "SwiftTypes.h"
#include "TensorProblem.h"

class InputParameters;

class SwiftConstantInterface
{
public:
  SwiftConstantInterface(const InputParameters & params);

  template <typename T>
  const T & getConstant(const std::string & param_name);
  template <typename T>
  const T & getConstantByName(const SwiftConstantName & name);

  template <typename T>
  void declareConstant(const std::string & param_name, const T & value);
  template <typename T>
  void declareConstantByName(const SwiftConstantName & name, const T & value);

protected:
  const InputParameters & _params;
  TensorProblem & _sci_tensor_problem;
};

template <typename T>
const T &
SwiftConstantInterface::getConstant(const std::string & param_name)
{
  return getConstantByName<T>(_params.get<SwiftConstantName>(param_name));
}

template <typename T>
const T &
SwiftConstantInterface::getConstantByName(const SwiftConstantName & name)
{
  return _sci_tensor_problem.getConstant<T>(name);
}

template <typename T>
void
SwiftConstantInterface::declareConstant(const std::string & param_name, const T & value)
{
  declareConstantByName<T>(_params.get<SwiftConstantName>(param_name), value);
}

template <typename T>
void
SwiftConstantInterface::declareConstantByName(const SwiftConstantName & name, const T & value)
{
  _sci_tensor_problem.declareConstant<T>(name, value);
}
