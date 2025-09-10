# LBMConstantTensor

!syntax description /TensorComputes/Solve/LBMConstantTensor

Fills the buffer with given constant value.

## Overview

Writes a constant scalar or vector into the target buffer each time it runs. Use
[!param](/TensorComputes/Solve/LBMConstantTensor/buffer) to select the destination and
[!param](/TensorComputes/Solve/LBMConstantTensor/constants) to provide one or more values
matching the buffer type and stencil dimension.

## Example Input File Syntax

!listing test/tests/lbm/channel2D.i block=TensorComputes/Initialize/initial_velocity

!syntax parameters /TensorComputes/Solve/LBMConstantTensor

!syntax inputs /TensorComputes/Solve/LBMConstantTensor

!syntax children /TensorComputes/Solve/LBMConstantTensor
