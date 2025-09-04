# LBMComputeForces

!syntax description /TensorComputes/Solve/LBMComputeForces

LBMComputeForces calculates forces for LBM simulations. Currently available forces are gravity and buoyancy.

## Overview

Computes body force contributions (e.g., pressure gradients or capillary forces) required by the
LBM forcing term. Supply required inputs (such as gradients or chemical potentials) and select the
output force buffer via [!param](/TensorComputes/Solve/LBMComputeForces/buffer).

## Example Input File Syntax

!listing test/tests/lbm/phase.i block=TensorComputes/Solve/forces

!syntax parameters /TensorComputes/Solve/LBMComputeForces

!syntax inputs /TensorComputes/Solve/LBMComputeForces

!syntax children /TensorComputes/Solve/LBMComputeForces
