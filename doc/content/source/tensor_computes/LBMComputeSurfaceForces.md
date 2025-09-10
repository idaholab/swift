# LBMComputeSurfaceForces

!syntax description /TensorComputes/Solve/LBMComputeSurfaceForces

This object acts similarly to LBMComputeForces and computes surface forces from chemical potential and gradient of phase-field order parameter.

## Overview

Computes interfacial tension force density from a chemical potential and gradient of phase field parameter for phase\-field LBM.
coupling. Supply the chemical potential with
[!param](/TensorComputes/Solve/LBMComputeSurfaceForces/chemical_potential) and the gradient with
[!param](/TensorComputes/Solve/LBMComputeSurfaceForces/grad_phi). The result is a vector field
written to [!param](/TensorComputes/Solve/LBMComputeSurfaceForces/buffer).

## Example Input File Syntax

!listing test/tests/lbm/phase.i block=TensorComputes/Solve/forces

!syntax parameters /TensorComputes/Solve/LBMComputeSurfaceForces

!syntax inputs /TensorComputes/Solve/LBMComputeSurfaceForces

!syntax children /TensorComputes/Solve/LBMComputeSurfaceForces
