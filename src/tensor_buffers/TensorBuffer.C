/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorBuffer.h"

template <typename T>
InputParameters
TensorBuffer<T>::validParams()
{
  InputParameters params = TensorBufferBase::validParams();
  return params;
}

template <typename T>
TensorBuffer<T>::TensorBuffer(const InputParameters & parameters) : TensorBufferBase(parameters)
{
}

template <typename T>
std::size_t
TensorBuffer<T>::advanceState()
{
  // make room to push state one step further back
  if (_u_old.size() < _max_states)
    _u_old.resize(_u_old.size() + 1);

  // push state further back
  if (!_u_old.empty())
  {
    for (std::size_t i = _u_old.size() - 1; i > 0; --i)
      _u_old[i] = _u_old[i - 1];
    _u_old[0] = _u;
  }

  return _u_old.size();
}

template <typename T>
void
TensorBuffer<T>::clearStates()
{
  _u_old.clear();
}

template <typename T>
void
TensorBuffer<T>::makeCPUCopy()
{
  if (_u.is_cpu())
    _u_cpu = _u.clone().contiguous();
  else
    _u_cpu = _u.cpu().contiguous();
}

template class TensorBuffer<torch::Tensor>;
