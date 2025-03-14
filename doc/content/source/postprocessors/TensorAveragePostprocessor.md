# TensorAveragePostprocessor

!syntax description /Postprocessors/TensorAveragePostprocessor

Takes the sum over all grid values divided by the number of grid cells (using `torch::sum` and `torch::numel`).

## Example Input File Syntax

!! Describe and include an example of how to use the TensorAveragePostprocessor object.

!syntax parameters /Postprocessors/TensorAveragePostprocessor

!syntax inputs /Postprocessors/TensorAveragePostprocessor

!syntax children /Postprocessors/TensorAveragePostprocessor
