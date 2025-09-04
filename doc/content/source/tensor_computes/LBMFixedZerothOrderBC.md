# LBMFixedZerothOrderBC9Q

!syntax description /TensorComputes/Solve/LBMFixedZerothOrderBC9Q

LBMFixedZerothOrderBC implements Zou-He pressure boundary conditions at the inlet and outlet for D2Q9, D3Q19 and D3Q27 stencils. The choice of stencil should be indicated in the name of the compute object such as LBMFixedZerothOrderBC9Q for D2Q9.

## Overview

Enforces zeroth\-order accurate macroscopic variables at selected domain faces. Choose faces with
[!param](/TensorComputes/Solve/LBMFixedZerothOrderBC9Q/boundary) and provide macroscopic fields as
required by the implementation.

## Example Input File Syntax

No minimal example is included yet.

!syntax parameters /TensorComputes/Solve/LBMFixedZerothOrderBC9Q

!syntax inputs /TensorComputes/Solve/LBMFixedZerothOrderBC9Q

!syntax children /TensorComputes/Solve/LBMFixedZerothOrderBC9Q
