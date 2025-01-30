/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorPostprocessor.h"
#include "TensorProblem.h"

template <class T>
InputParameters
TensorPostprocessorTempl<T>::validParams()
{
  InputParameters params = T::validParams();
  params.addClassDescription("A normal Postprocessor acting on a Tensor buffer.");
  params.addRequiredParam<TensorInputBufferName>("buffer", "The buffer this compute is operating on");
  return params;
}

template <class T>
TensorPostprocessorTempl<T>::TensorPostprocessorTempl(const InputParameters & parameters)
  : T(parameters),
    DomainInterface(this),
    _tensor_problem(
        [this]()
        {
          auto tensor_problem = dynamic_cast<TensorProblem *>(&this->_fe_problem);
          if (!tensor_problem)
            mooseError("TensorPostprocessors require a TensorProblem.");
          return std::ref(*tensor_problem);
        }()),
    _u(_tensor_problem.getBuffer(this->template getParam<TensorInputBufferName>("buffer")))
{
}

template class TensorPostprocessorTempl<GeneralPostprocessor>;
template class TensorPostprocessorTempl<GeneralVectorPostprocessor>;
