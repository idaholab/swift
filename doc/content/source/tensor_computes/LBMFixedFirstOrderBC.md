# LBMFixedFirstOrderBC9Q

!syntax description /TensorComputes/Solve/LBMFixedFirstOrderBC9Q

LBMFixedFirstOrderBC implements Zou\-He velocity boundary conditions at the inlet and outlet for D2Q9, D3Q19 and D3Q27 stencils. The choice of stencil should be indicated in the name of the compute object such as LBMFixedFirstOrderBC9Q for D2Q9.

## Overview

Enforces first\-order accurate macroscopic velocity at selected domain faces via Zou\-He formulas.
Choose faces with [!param](/TensorComputes/Solve/LBMFixedFirstOrderBC9Q/boundary) and provide
macroscopic fields as required by the implementation.

## Example Input File Syntax

!listing test/tests/lbm/vertical_velocity_bcs.i block=TensorComputes/Boundary/top
!listing test/tests/lbm/vertical_velocity_bcs.i block=TensorComputes/Boundary/bottom

!syntax parameters /TensorComputes/Solve/LBMFixedFirstOrderBC9Q

!syntax inputs /TensorComputes/Solve/LBMFixedFirstOrderBC9Q

!syntax children /TensorComputes/Solve/LBMFixedFirstOrderBC9Q
