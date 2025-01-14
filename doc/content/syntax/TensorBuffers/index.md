# TensorBuffers System

The solution fields (or tensors) are declared under the `TensorBuffers` top-level block.
Tensors can be thought of as the equivalent concept to `Variables` in MOOSE. Tensors exist on fixed regular grids
of dimension 2 or 3, as defined in the [Domain](Domain/index.md) block.

## Example Input File Syntax

!syntax list /TensorBuffers objects=True actions=False subsystems=False

!syntax list /TensorBuffers objects=False actions=False subsystems=True

!syntax list /TensorBuffers objects=False actions=True subsystems=False
