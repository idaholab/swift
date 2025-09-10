# TensorIntegralChangePostprocessor

!alert construction title=Undocumented Class
The TensorIntegralChangePostprocessor has not been documented. The content listed below should be used as a starting point for
documenting the class, which includes the typical automatic documentation associated with a
MooseObject; however, what is contained is ultimately determined by what is necessary to make the
documentation clear for users.

!syntax description /UserObjects/TensorIntegralChangePostprocessor

## Overview

Reports the change in the integral of a buffer between successive executions, useful for monitoring
mass conservation or convergence. Select the input with
[!param](/Postprocessors/TensorIntegralChangePostprocessor/buffer).

## Example Input File Syntax

!listing test/tests/cahnhilliard/cahnhilliard.i block=Postprocessors/delta_int_c

!syntax parameters /UserObjects/TensorIntegralChangePostprocessor

!syntax inputs /UserObjects/TensorIntegralChangePostprocessor

!syntax children /UserObjects/TensorIntegralChangePostprocessor
