# LBMSmagorinskyMRTCollision

!syntax description /TensorComputes/Solve/LBMSmagorinskyMRTCollision

This objects implements commonly used collision dynamics for Lattice Boltzmann simulation. Currently available collision operators are BGK single relaxation time and Multi Relaxation Time (MRT) operators. The additional operation is available to project the non-equilibrium distribution onto Hermite space to achieve stability with the boolean parameter `projection`. For high Reynolds number simulations, Smagorinsky LES collision dynamics is available for both BGK and MRT operators.

## Overview

Template for multiple LBM collision operators:

- BGK: single\-relaxation time model (`LBMBGKCollision`).
- MRT: multi\-relaxation time in moment space (`LBMMRTCollision`).
- Smagorinsky: eddy\-viscosity based turbulence (`LBMSmagorinskyCollision`, `LBMSmagorinskyMRTCollision`).

Supply incoming and equilibrium distributions via
[!param](/TensorComputes/Solve/LBMBGKCollision/f) and
[!param](/TensorComputes/Solve/LBMBGKCollision/feq). Relaxation is controlled with
[!param](/TensorComputes/Solve/LBMBGKCollision/tau0); Smagorinsky models also use
[!param](/TensorComputes/Solve/LBMBGKCollision/Cs).

## Example Input File Syntax

!listing test/tests/lbm/channel2D.i block=TensorComputes/Solve/collision

!syntax parameters /TensorComputes/Solve/LBMSmagorinskyMRTCollision

!syntax inputs /TensorComputes/Solve/LBMSmagorinskyMRTCollision

!syntax children /TensorComputes/Solve/LBMSmagorinskyMRTCollision
