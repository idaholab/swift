# Initialize System

Tensor operators under this system are executed at the start of the simulation to set up initial values of
tensors. Dependency resolution is applied to automatically sort the execution order of dependent operators.

## Example Input File Syntax

!! Describe and include an example of how to use the Initialize system.

!syntax list /TensorComputes/Initialize objects=True actions=False subsystems=False

!syntax list /TensorComputes/Initialize objects=False actions=False subsystems=True

!syntax list /TensorComputes/Initialize objects=False actions=True subsystems=False
