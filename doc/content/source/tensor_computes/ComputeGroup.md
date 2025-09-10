# ComputeGroup

!syntax description /TensorComputes/Solve/ComputeGroup

Combines multiple compute operators into one execution unit and applies dependency resolution among
them. Compute groups may be nested. Inputs of a group are all inputs of the members that are not
produced by any member; outputs are all members' outputs not consumed by any member. Provide the
members with [!param](/TensorComputes/Solve/ComputeGroup/computes).

## Example Input File Syntax

!listing test/tests/tensor_compute/group.i block=TensorComputes/Solve/group

!syntax parameters /TensorComputes/Solve/ComputeGroup

!syntax inputs /TensorComputes/Solve/ComputeGroup

!syntax children /TensorComputes/Solve/ComputeGroup
