# LBMFixedZerothOrderBC9Q

!syntax description /TensorComputes/Solve/LBMFixedZerothOrderBC9Q

LBMFixedZerothOrderBC implements Zou-He pressure boundary conditions at the inlet and outlet for D2Q9, D3Q19 and D3Q27 stencils. The choice of stencil should be indicated in the name of the compute object such as LBMFixedZerothOrderBC9Q for D2Q9.

## Overview

LBMFixedZerothOrderBC implements Zou\-He pressure boundary conditions at the inlet and outlet for D2Q9, D3Q19 and D3Q27 stencils. The choice of stencil should be indicated in the name of the compute object such as LBMFixedZerothOrderBC9Q for D2Q9.
Choose faces with [!param](/TensorComputes/Solve/LBMFixedZerothOrderBC9Q/boundary) and provide macroscopic fields as
required by the implementation.

## Example Input File Syntax

!listing test/tests/lbm/vertical_density_bcs.i block=TensorComputes/Boundary/top
!listing test/tests/lbm/vertical_density_bcs.i block=TensorComputes/Boundary/bottom

!syntax parameters /TensorComputes/Solve/LBMFixedZerothOrderBC9Q

!syntax inputs /TensorComputes/Solve/LBMFixedZerothOrderBC9Q

!syntax children /TensorComputes/Solve/LBMFixedZerothOrderBC9Q
