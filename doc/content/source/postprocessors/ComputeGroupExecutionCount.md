# ComputeGroupExecutionCount

!syntax description /Postprocessors/ComputeGroupExecutionCount

Returns the number of times a specified `ComputeGroup` has executed its
`computeBuffer()` method. This is useful for debugging and verifying that a
group of tensor computes runs the expected number of times during a solve.

- Default target: `compute_group = root`.
- Scope: counts cumulative executions since the start of the run.

## Example Input File Syntax

The example below reports how many times the top-level compute group `root`
executed during the solve.

```
[Postprocessors]
  [count]
    type = ComputeGroupExecutionCount
    # Optional: choose a specific group name
    # compute_group = root
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]
```

To target a different group, set `compute_group` to the name of that
`ComputeGroup` defined under `[TensorComputes/Solve]`.

!syntax parameters /Postprocessors/ComputeGroupExecutionCount

!syntax inputs /Postprocessors/ComputeGroupExecutionCount

!syntax children /Postprocessors/ComputeGroupExecutionCount

