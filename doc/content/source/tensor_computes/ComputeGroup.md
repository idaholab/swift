# ComputeGroup

This operator combines multiple specified operators as one execution unit and applies dependency resolution
for the grouped operators. Compute groups may be nested.

Inputs of a compute group are all required inputs of the contained operators that are not provided as outputs by any contained operator. Conversely outputs of the compute group are all outputs of the contained operators that are not
consumed as inputs by any contained operator.

## Example Input File Syntax

!! Describe and include an example of how to use the ComputeGroup object.

!syntax parameters /TensorComputes/Solve/ComputeGroup

!syntax inputs /TensorComputes/Solve/ComputeGroup

!syntax children /TensorComputes/Solve/ComputeGroup
