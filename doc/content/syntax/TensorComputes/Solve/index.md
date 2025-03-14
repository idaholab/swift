# Solve System

Tensor operators under this system are executed by the [TensorSolver](TensorSolver/index.md) during the solver iterations.
Dependency resolution is applied to automatically sort the execution order of dependent operators.

## Example Input File Syntax

!! Describe and include an example of how to use the Solve system.

!syntax list /TensorComputes/Solve objects=True actions=False subsystems=False

!syntax list /TensorComputes/Solve objects=False actions=False subsystems=True

!syntax list /TensorComputes/Solve objects=False actions=True subsystems=False
