/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * ReciprocalMatDiffusion (\nabla M \nabla \mu) object
 */

 class ReciprocalMatDiffusion : public TensorOperator
 {
public:
    static InputParameters validParams();

    ReciprocalMatDiffusion(const InputParameters & parameters);

    virtual void computeBuffer() override;

protected:
    const torch::Tensor & _chem_pot;
    const torch::Tensor & _M;
    const torch::Tensor & _psi;
    const Real _epsilon;

    /// imaginary unit i
    const torch::Tensor _imag;
};