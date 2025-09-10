# TensorAveragePostprocessor

!syntax description /Postprocessors/TensorAveragePostprocessor

Takes the sum over all grid values divided by the number of grid cells (using `torch::sum` and `torch::numel`).

## Overview

Computes the spatial average of a scalar buffer over the domain. Select the input with
[!param](/Postprocessors/TensorAveragePostprocessor/buffer).

## Example Input File Syntax

!listing test/tests/postprocessors/postprocessors.i block=Postprocessors/avg_c

!syntax parameters /Postprocessors/TensorAveragePostprocessor

!syntax inputs /Postprocessors/TensorAveragePostprocessor

!syntax children /Postprocessors/TensorAveragePostprocessor
