# TensorExtremeValuePostprocessor

!syntax description /Postprocessors/TensorExtremeValuePostprocessor

Finds the largest (`torch::max`) or smallest (`torch::min`) value in the given buffer.

## Overview

Computes the minimum or maximum over a scalar buffer. Choose the input with
[!param](/Postprocessors/TensorExtremeValuePostprocessor/buffer) and the operation via
[!param](/Postprocessors/TensorExtremeValuePostprocessor/value_type) set to `MIN` or `MAX`.

## Example Input File Syntax

!listing test/tests/tensor_compute/group.i block=Postprocessors/max_c

!syntax parameters /Postprocessors/TensorExtremeValuePostprocessor

!syntax inputs /Postprocessors/TensorExtremeValuePostprocessor

!syntax children /Postprocessors/TensorExtremeValuePostprocessor
