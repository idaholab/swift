# OnDemand System

Tensor operators under this system are not executed by the solver directly, instead they are made available by the `getOnDemandCompute(name)` API and can be run explicitly as nested computes by other operators (like MOOSE `compute=false` materials). If dependency resolution of dependent OnDemand operators is required they should be grouped using [ComputeGroup](ComputeGroup.md) operators.

## Example Input File Syntax

!! Describe and include an example of how to use the OnDemand system.

!syntax list /TensorComputes/OnDemand objects=True actions=False subsystems=False

!syntax list /TensorComputes/OnDemand objects=False actions=False subsystems=True

!syntax list /TensorComputes/OnDemand objects=False actions=True subsystems=False
