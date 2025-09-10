# LBMEquilibrium

!syntax description /TensorComputes/Solve/LBMEquilibrium

Compute LBM equilibrium distribution functions. The equilibrium uses a second-order Hermite
expansion of the Maxwell-Boltzmann distribution.

## Overview

Given bulk scalar `bulk` (e.g., density or temperature) and velocity vector field `velocity`, this
object computes `f_eq` on all lattice directions using the stencil weights and the lattice sound
speed `c_s` defined by the `LatticeBoltzmannProblem`.

## Example Input File Syntax

!listing
[TensorComputes]
  [Solve]
    [feq]
      type = LBMEquilibrium
      buffer = feq
      bulk = rho
      velocity = u
    []
  []
[]
!listing-end

!syntax parameters /TensorComputes/Solve/LBMEquilibrium

!syntax inputs /TensorComputes/Solve/LBMEquilibrium

!syntax children /TensorComputes/Solve/LBMEquilibrium
