# ReciprocalMatDiffusion

!syntax description /TensorComputes/Solve/ReciprocalMatDiffusion

ReciprocalMatDiffusion computes the divergence of the flux `\mathbf{J} = - M \, \nabla \mu` with variable mobility `M` using FFTs, and adds a no\-flux correction on masked boundaries `psi>0`.

## Overview

Inputs:
- `chemical_potential`: field `\mu`.
- `mobility`: mobility `M(\cdot)`.
- `psi`: mask for imposing Neumann no\-flux boundaries. Set `always_update_psi = true` if the mask changes.

The object computes `\nabla \cdot \mathbf{J}` in reciprocal space and adds `\nabla psi / psi \cdot \mathbf{J}` inside the mask to enforce `\mathbf{n} \cdot \mathbf{J} = 0`.

## Example Input File Syntax

!listing test/tests/kks/KKS_no_flux_bc.i block=TensorComputes/Solve/kappa_grad_eta

!syntax parameters /TensorComputes/Solve/ReciprocalMatDiffusion

!syntax inputs /TensorComputes/Solve/ReciprocalMatDiffusion

!syntax children /TensorComputes/Solve/ReciprocalMatDiffusion

