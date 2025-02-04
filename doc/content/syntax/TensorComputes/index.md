# TensorComputes System

`TensorComputes` is the parent system of the following subsystems

* [Initialize](Solve/index.md)
* [Solve](Solve/index.md)
* [OnDemand](Solve/index.md)
* [Postprocess](Solve/index.md)

Each one of those systems holds a set of dependency resolved Tensor operators  that are executed at
different stages of the simulation.

## Example Input File Syntax

!! Describe and include an example of how to use the TensorComputes system.

!syntax list /TensorComputes objects=True actions=False subsystems=False

!syntax list /TensorComputes objects=False actions=False subsystems=True

!syntax list /TensorComputes objects=False actions=True subsystems=False
