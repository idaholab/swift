# LBMMacroscopicNeumannBC

!syntax description /TensorComputes/Solve/LBMMacroscopicNeumannBC

This object implements Neumann BC on macroscopic tensor buffers on left, right, top, bottom, front and back boundaries.

## Overview

Applies a Neumann\-type flux on a macroscopic scalar by adjusting distributions on selected faces.
Provide the target with [!param](/TensorComputes/Solve/LBMMacroscopicNeumannBC/buffer) and the flux
value via [!param](/TensorComputes/Solve/LBMMacroscopicNeumannBC/value). Choose faces with
[!param](/TensorComputes/Solve/LBMMacroscopicNeumannBC/boundary).

## Example Input File Syntax

!listing test/tests/lbm/neumann_box.i block=TensorComputes/Boundary/left

!syntax parameters /TensorComputes/Solve/LBMMacroscopicNeumannBC

!syntax inputs /TensorComputes/Solve/LBMMacroscopicNeumannBC

!syntax children /TensorComputes/Solve/LBMMacroscopicNeumannBC
