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

  virtual std::size_t advanceState();
  virtual void clearStates();
  virtual void makeCPUCopy();

  /// current state of the tensor
  T _u;

  /// potential CPU copy of the tensor (if requested)
  T _u_cpu;

  /// old states of the tensor
  std::vector<T> _u_old;
  std::size_t _max_states;
};
