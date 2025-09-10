# LBMComputeVelocity

!syntax description /TensorComputes/Solve/LBMComputeVelocity

LBMComputeVelocity computes macroscopic velocity from distribution functions. Forces can be added to velocity.

## Overview

Computes velocity from the distribution `f` and density `rho` using the stencil direction vectors
`e_x,e_y,e_z`:

`u_x = (1/rho) sum_q f e_{x,q}`, with analogous expressions for `u_y` and `u_z` in 2D/3D.

Optional force contributions are added as `(forces)/(2 rho)`; constant body forces can be enabled on
each axis via `add_body_force = true` and `body_force_{x,y,z}` constants.

## Example Input File Syntax

!listing
[TensorComputes]
  [Solve]
    [rho]
      type = LBMComputeDensity
      buffer = rho
      f = f
    []
    [vel]
      type = LBMComputeVelocity
      buffer = u
      f = f
      rho = rho
      enable_forces = true
      forces = F
      add_body_force = true
      body_force_x = '1e-6'
      body_force_y = '0.0'
      body_force_z = '0.0'
    []
  []
[]
!listing-end

!syntax parameters /TensorComputes/Solve/LBMComputeVelocity

!syntax inputs /TensorComputes/Solve/LBMComputeVelocity

!syntax children /TensorComputes/Solve/LBMComputeVelocity
