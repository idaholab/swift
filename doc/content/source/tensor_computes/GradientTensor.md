# GradientTensor

!syntax description /TensorComputes/Solve/GradientTensor

GradientTensor computes the spatial gradient of a scalar buffer and returns a vector tensor (`neml2::Vec`) with components `(\partial_x u, \partial_y u, \partial_z u)`. When [!param](/TensorComputes/Solve/GradientTensor/input_is_reciprocal) `= true`, the input is treated as already in reciprocal space.

## Overview

The gradient is computed spectrally using FFTs as `\nabla u = \mathcal{F}^{-1}\{ i \, \mathbf{k} \, \mathcal{F}\{u\}\}`. In 2D/1D, the unused components are zero.

Requires NEML2 support to enable the `neml2::Vec` value type.

## Example Input File Syntax

!listing test/tests/typed_tensors/gradient.i block=TensorComputes/Initialize/grad_c

!syntax parameters /TensorComputes/Solve/GradientTensor

!syntax inputs /TensorComputes/Solve/GradientTensor

!syntax children /TensorComputes/Solve/GradientTensor

