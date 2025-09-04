# LBMTensorBuffer

!syntax description /TensorBuffers/LBMTensorBuffer

This object provides common tensor buffers for Lattice Boltzmann simulations. `df` type buffer adds distribution functions with extra dimension of size Q which corresponds to the choice of stencil. `mv` type buffer adds macroscopic vectorial buffers such as velocity and force. `ms` type buffer adds macroscopic scalar buffers such as density and speed.

## Overview

Holds lattice quantities for LBM simulations: distribution functions (df), macroscopic scalars (ms)
and vectors (mv). Use [!param](/TensorBuffers/LBMTensorBuffer/buffer_type) to choose the storage
type and [!param](/TensorBuffers/LBMTensorBuffer/buffer) to name the buffer.

## Example Input File Syntax

!listing test/tests/lbm/channel2D.i block=TensorBuffers/velocity

!syntax parameters /TensorBuffers/LBMTensorBuffer

!syntax inputs /TensorBuffers/LBMTensorBuffer

!syntax children /TensorBuffers/LBMTensorBuffer
