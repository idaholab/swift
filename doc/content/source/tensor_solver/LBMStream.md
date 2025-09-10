# LBMStream

!syntax description /TensorSolver/LBMStream

This tensor solver moves LBM distributions in time by streaming them based to the chosen stencil. The old state buffers must be post collision distributions. Multiple input and output buffers can be provided.

## Overview

Streams distribution functions along stencil directions to advance in time. Provide the active
distribution buffer via [!param](/TensorSolver/LBMStream/buffer) and the post\-collision history via
[!param](/TensorSolver/LBMStream/f_old).

## Example Input File Syntax

!listing test/tests/lbm/channel2D.i block=TensorSolver

!syntax parameters /TensorSolver/LBMStream

!syntax inputs /TensorSolver/LBMStream

!syntax children /TensorSolver/LBMStream
