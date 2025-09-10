# LBMIsotropicGradient

!syntax description /TensorComputes/Solve/LBMIsotropicGradient

This object uses isotropic finite difference method ot compute the gradient of phase field order parameter for LBM simulations.

## Overview

Computes an isotropic finite\-difference approximation to $\nabla \phi$ on the LBM grid. Provide
the scalar field with
[!param](/TensorComputes/Solve/LBMIsotropicGradient/scalar_field) and select the destination
vector buffer via [!param](/TensorComputes/Solve/LBMIsotropicGradient/buffer).

## Example Input File Syntax

!listing test/tests/lbm/phase.i block=TensorComputes/Solve/grad_phi

!syntax parameters /TensorComputes/Solve/LBMIsotropicGradient

!syntax inputs /TensorComputes/Solve/LBMIsotropicGradient

!syntax children /TensorComputes/Solve/LBMIsotropicGradient
