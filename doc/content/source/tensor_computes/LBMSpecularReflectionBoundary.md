# LBMSpecularReflectionBoundary

!syntax description /TensorComputes/Solve/LBMSpecularReflectionBoundary

Implements a linear combination of bounce-back and specular-reflection wall treatment for LBM.
Incoming distributions are split between opposite directions (bounce-back) and mirror directions
(specular) using mixing ratio [!param](/TensorComputes/Solve/LBMSpecularReflectionBoundary/r).
The previous time step distributions are supplied via
[!param](/TensorComputes/Solve/LBMSpecularReflectionBoundary/f_old).

!alert warning title=Experimental
This boundary condition is under development and not yet tested.

!syntax parameters /TensorComputes/Solve/LBMSpecularReflectionBoundary

!syntax inputs /TensorComputes/Solve/LBMSpecularReflectionBoundary

!syntax children /TensorComputes/Solve/LBMSpecularReflectionBoundary

