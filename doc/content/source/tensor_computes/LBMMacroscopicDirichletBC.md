# LBMMacroscopicDirichletBC

!syntax description /TensorComputes/Solve/LBMMacroscopicDirichletBC

This object applies Dirichlet boundary conditions on macroscopic buffers at the domain faces
(`left`, `right`, `top`, `bottom`, `front`, `back`). For solid boundaries spanning interior cells,
use [LBMDirichletWallBC](LBMDirichletWallBC.md).

## Overview

Sets the target buffer to the constant value `value` on the selected faces. The `wall` boundary is
not supported here; see `LBMDirichletWallBC` for that use case.

## Example Input File Syntax

!listing
[TensorComputes]
  [Solve]
    [inlet_outlet]
      type = LBMMacroscopicDirichletBC
      buffer = rho
      boundary = 'left right'
      value = '1.0'
    []
  []
[]
!listing-end

!syntax parameters /TensorComputes/Solve/LBMMacroscopicDirichletBC

!syntax inputs /TensorComputes/Solve/LBMMacroscopicDirichletBC

!syntax children /TensorComputes/Solve/LBMMacroscopicDirichletBC
