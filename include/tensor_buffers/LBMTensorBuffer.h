/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include"TensorBuffer.h"

/**
 * Tensor wrapper for LBM tensors
 */
class LBMTensorBuffer : public TensorBuffer<torch::Tensor>
{
public:
    static InputParameters validParams();

    LBMTensorBuffer(const InputParameters & parameters);

    void init();

protected:
    Real _dimension;
};

// registerTensorType(LBMTensorBuffer, torch::Tensor);
