# LBMIsotropicLaplacian

!syntax description /TensorComputes/Solve/LBMIsotropicLaplacian

This object uses isotropic finite difference method ot compute the Lapcian of phase field order parameter for LBM simulations.

## Overview

Computes an isotropic finite\-difference approximation to the Laplacian $\nabla^2 \phi$ on the
LBM grid. Provide the scalar field with
[!param](/TensorComputes/Solve/LBMIsotropicLaplacian/scalar_field) and select the destination
scalar buffer via [!param](/TensorComputes/Solve/LBMIsotropicLaplacian/buffer).

## Example Input File Syntax

!listing test/tests/lbm/phase.i block=TensorComputes/Solve/laplacian_phi

!syntax parameters /TensorComputes/Solve/LBMIsotropicLaplacian

!syntax inputs /TensorComputes/Solve/LBMIsotropicLaplacian

!syntax children /TensorComputes/Solve/LBMIsotropicLaplacian
