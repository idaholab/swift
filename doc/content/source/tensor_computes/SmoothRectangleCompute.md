# ParsedCompute

!syntax description /TensorComputes/Solve/SmoothRectangleCompute

## Example Input File Syntax

SmoothRectangleCompute allows interpolating a tensor value inside and outside of a specified box. The box is aligned with the x, y, z axes and is specified by passing in the x, y, z coordinates of the bottom left point and the top right point. Each of the coordinates of the "bottom_left" point MUST be less than those coordinates in the "top_right" point.

When setting the initial condition, if bottom_left <= Point <= top_right then the "inside" value is used. Otherwise the "outside" value is used. The `profile` parameter sets whether the interpolation uses a COS or TANH interpolation function. Setting the interface width to 0 allows for sharp interpolation, though this may cause numerical issues with the FFT solver.

!syntax parameters /TensorComputes/Solve/SmoothRectangleCompute

!syntax inputs /TensorComputes/Solve/SmoothRectangleCompute

!syntax children /TensorComputes/Solve/SmoothRectangleCompute
