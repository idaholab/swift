# FFTQuasistaticElasticity

!syntax description /TensorComputes/Solve/FFTQuasistaticElasticity

## Overview

Solves a homogeneous, linear, quasi-static elasticity system in Fourier space using Lam\'e
parameters `mu` and `lambda` and an eigenstrain amplitude `e0` driven by a scalar field `cbar`
provided in reciprocal space. The object solves for displacements `u = (u_x, u_y, u_z)` and writes
them to the output buffers listed in `displacements` (one per dimension), by assembling and solving
the spectral linear system `A(\vec k) \hat u = b(\vec k)` and then inverse transforming.

## Example Input File Syntax

!listing
[TensorComputes]
  [Solve]
    [u_quasi]
      type = FFTQuasistaticElasticity
      # outputs: one buffer per dimension
      displacements = 'ux uy uz'
      # inputs
      cbar = cbar
      mu = 1.0
      lambda = 2.0
      e0 = 1e-3
    []
  []
[]
!listing-end

See also: [ForwardFFT](PerformFFT.md), [InverseFFT](PerformFFT.md),
[FFTElasticChemicalPotential](FFTElasticChemicalPotential.md).

!syntax parameters /TensorComputes/Solve/FFTQuasistaticElasticity

!syntax inputs /TensorComputes/Solve/FFTQuasistaticElasticity

!syntax children /TensorComputes/Solve/FFTQuasistaticElasticity
