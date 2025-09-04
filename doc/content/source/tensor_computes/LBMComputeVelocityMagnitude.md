# LBMComputeVelocityMagnitude

!syntax description /TensorComputes/Solve/LBMComputeVelocityMagnitude

Computes the magnitude of velocity (speed) from velocity buffer.

## Overview

Computes the Euclidean magnitude of a velocity vector field and stores it into the output buffer.
Select the destination via [!param](/TensorComputes/Solve/LBMComputeVelocityMagnitude/buffer)
and supply the vector field with
[!param](/TensorComputes/Solve/LBMComputeVelocityMagnitude/velocity).

## Example Input File Syntax

!listing test/tests/lbm/channel2D.i block=TensorComputes/Solve/speed

!syntax parameters /TensorComputes/Solve/LBMComputeVelocityMagnitude

!syntax inputs /TensorComputes/Solve/LBMComputeVelocityMagnitude

!syntax children /TensorComputes/Solve/LBMComputeVelocityMagnitude
