# LBMDirichletWallBC

!syntax description /TensorComputes/Solve/LBMDirichletWallBC

This LBMDirichletWallBC objects implements perliminary version of Dirichlet boundary condition at the solid walls.

## Overview

Imposes a fixed value on a macroscopic field at solid walls by reconstructing incoming
distributions to enforce the desired boundary condition. Select the target field via
[!param](/TensorComputes/Solve/LBMDirichletWallBC/buffer) and choose faces using
[!param](/TensorComputes/Solve/LBMDirichletWallBC/boundary).

## Example Input File Syntax

!listing test/tests/lbm/advan_bc.i block=TensorComputes/Boundary/bottom

!syntax parameters /TensorComputes/Solve/LBMDirichletWallBC

!syntax inputs /TensorComputes/Solve/LBMDirichletWallBC

!syntax children /TensorComputes/Solve/LBMDirichletWallBC
