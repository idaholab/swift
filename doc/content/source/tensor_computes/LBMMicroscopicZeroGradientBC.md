# LBMMicroscopicZeroGradientBC

!syntax description /TensorComputes/Solve/LBMMicroscopicZeroGradientBC

This object implements zero flux Neumann BC on LBM distribution functions.

## Overview

Imposes zero normal gradient on the microscopic distributions at selected domain faces. Choose
faces via [!param](/TensorComputes/Solve/LBMMicroscopicZeroGradientBC/boundary) and provide the
target distribution via [!param](/TensorComputes/Solve/LBMMicroscopicZeroGradientBC/buffer).

## Example Input File Syntax

!listing test/tests/lbm/obstacle.i block=TensorComputes/Boundary/right

!syntax parameters /TensorComputes/Solve/LBMMicroscopicZeroGradientBC

!syntax inputs /TensorComputes/Solve/LBMMicroscopicZeroGradientBC

!syntax children /TensorComputes/Solve/LBMMicroscopicZeroGradientBC
