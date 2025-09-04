# LBMStackTensors

!syntax description /TensorComputes/Solve/LBMStackTensors

Stacks multiple scalar tensors to create one vectorial tensor buffer.

## Overview

Stacks multiple scalar or vector buffers along the stencil index to form a distribution\-like
tensor with `Q` components per cell. Useful for constructing custom distribution arrays from
component fields.

## Example Input File Syntax

!listing test/tests/lbm/isotropic_stencil_mrt.i block=TensorComputes/Initialize/stack

!syntax parameters /TensorComputes/Solve/LBMStackTensors

!syntax inputs /TensorComputes/Solve/LBMStackTensors

!syntax children /TensorComputes/Solve/LBMStackTensors
