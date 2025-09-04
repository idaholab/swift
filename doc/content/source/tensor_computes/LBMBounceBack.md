# LBMBounceBack

!syntax description /TensorComputes/Solve/LBMBounceBack

This compute object implements simple bounce-back rule on boundaries for Lattice Boltzmann simulations. Boundary can be on the left, right, top, bottom, front and back as well as wall. Wall boundary referes to any solid object the fluid cannot penetrate.

## Overview

Imposes no-penetration by reflecting incoming distributions into their opposite directions at the
selected boundary. Supports domain faces (`left`, `right`, `top`, `bottom`, `front`, `back`) and
`wall` for solid-embedded geometries. For 3D binary media masks, adjacent-to-solid cells are handled
by a specialized path. Corner exclusion on each axis can be enabled to avoid double-applying rules.

## Example Input File Syntax

!listing
[TensorComputes]
  [Solve]
    [bb]
      type = LBMBounceBack
      buffer = f
      f_old = f
      boundary = 'left right top bottom'
      exclude_corners_x = true
    []
  []
[]
!listing-end

!syntax parameters /TensorComputes/Solve/LBMBounceBack

!syntax inputs /TensorComputes/Solve/LBMBounceBack

!syntax children /TensorComputes/Solve/LBMBounceBack
