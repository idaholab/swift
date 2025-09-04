# FFTElasticChemicalPotential

!syntax description /TensorComputes/Solve/FFTElasticChemicalPotential

## Overview

Computes the elastic contribution to the chemical potential in Fourier space for an eigenstrain
problem. Given Lam\'e parameters `mu` and `lambda`, an eigenstrain amplitude `e0`, the Fourier
transform of a scalar field `cbar`, and the displacement fields `u = (u_x,u_y,u_z)`, the output is

`\mu_\text{mech} = -e0 * ( e0 * (9 lambda cbar + 6 mu cbar) - (2 mu + 3 lambda) * div(u) )`,

where `div(u)` is evaluated spectrally as `i 2 pi (k_x \hat u_x + k_y \hat u_y + k_z \hat u_z)`.

The resulting tensor is inverse-transformed by downstream operators as needed.

## Example Input File Syntax

!listing
[TensorComputes]
  [Solve]
    [elastic_mu]
      type = FFTElasticChemicalPotential
      buffer = mubar_elastic
      displacements = 'ux uy uz'
      cbar = cbar
      mu = 1.0
      lambda = 2.0
      e0 = 1e-3
    []
  []
[]
!listing-end

See also: [FFTQuasistaticElasticity](FFTQuasistaticElasticity.md), [ForwardFFT](PerformFFT.md).

!syntax parameters /TensorComputes/Solve/FFTElasticChemicalPotential

!syntax inputs /TensorComputes/Solve/FFTElasticChemicalPotential

!syntax children /TensorComputes/Solve/FFTElasticChemicalPotential
