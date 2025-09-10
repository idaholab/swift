# TensorIntegralPostprocessor

!syntax description /Postprocessors/TensorIntegralPostprocessor

Takes the sum over all grid values multiplied by the grid cell volume.

## Overview

Integrates the values in a scalar buffer over the domain by summing all entries and multiplying by
the cell volume. Select the input with
[!param](/Postprocessors/TensorIntegralPostprocessor/buffer).

## Example Input File Syntax

!listing test/tests/postprocessors/postprocessors.i block=Postprocessors/int_c

!syntax parameters /Postprocessors/TensorIntegralPostprocessor

!syntax inputs /Postprocessors/TensorIntegralPostprocessor

!syntax children /Postprocessors/TensorIntegralPostprocessor
