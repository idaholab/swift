/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorBufferBase.h"

/**
 * Tensor wrapper arbitrary tensor value dimensions
 */
template <typename T>
class TensorBuffer : public TensorBufferBase
{
public:
  static InputParameters validParams();

  TensorBuffer(const InputParameters & parameters);

  virtual std::size_t advanceState() override;
  virtual void clearStates() override;
  virtual void makeCPUCopy() override;

  T & getTensor();
  const std::vector<T> & getOldTensor(std::size_t states_requested);

  virtual const torch::Tensor & getRawTensor() const override;
  virtual const torch::Tensor & getRawCPUTensor() override;

protected:
  /// current state of the tensor
  T _u;

  /// potential CPU copy of the tensor (if requested)
  T _u_cpu;

  /// was a CPU copy requested?
  bool _cpu_copy_requested;

  /// old states of the tensor
  std::vector<T> _u_old;
  std::size_t _max_states;
};

template <typename T>
T &
TensorBuffer<T>::getTensor()
{
  return _u;
}

template <typename T>
const std::vector<T> &
TensorBuffer<T>::getOldTensor(std::size_t states_requested)
{
  _max_states = std::max(_max_states, states_requested);
  return _u_old;
}

/**
 * Specialization of this helper struct can be used to force the use of derived
 * classes for implicit TensorBuffer construction (i.e. tensors that are not explicitly
 * listed under [TensorBuffers]).
 */
template <typename T>
struct TensorBufferSpecialization
{
  using type = TensorBuffer<T>;
};

#define registerTensorType(derived_class, tensor_type)                                             \
  template <>                                                                                      \
  struct TensorBufferSpecialization<tensor_type>                                                   \
  {                                                                                                \
    using type = derived_class;                                                                    \
  }
