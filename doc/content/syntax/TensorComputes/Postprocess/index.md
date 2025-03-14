# Postprocess System

Tensor operators under this system are executed at the end of a MOOSE timestep, right before output occurs.
Posprocess operators allow for the computation of outputted postprocessed fields, like for example a Fourier
filtered version of a microstructure. Dependency resolution is applied to automatically sort the execution
order of dependent operators.

## Overview

!! Replace this line with information regarding the Postprocess system.

## Example Input File Syntax

!! Describe and include an example of how to use the Postprocess system.

!syntax list /TensorComputes/Postprocess objects=True actions=False subsystems=False

!syntax list /TensorComputes/Postprocess objects=False actions=False subsystems=True

!syntax list /TensorComputes/Postprocess objects=False actions=True subsystems=False
