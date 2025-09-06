# ComputeVonMisesStress

!syntax description /TensorComputes/Solve/ComputeVonMisesStress

ComputeVonMisesStress computes the scalar von Mises equivalent stress from a rank\-2 stress tensor field. Formulas for 2D and 3D are supported.

## Overview

Given Cauchy [!param](/TensorComputes/Solve/ComputeVonMisesStress/stress) components $\sigma_{ij}$, the 3D equivalent stress is

\begin{equation}
\sigma_v = \sqrt{\tfrac{1}{2} \big[(\sigma_{xx}-\sigma_{yy})^2 + (\sigma_{yy}-\sigma_{zz})^2 + (\sigma_{zz}-\sigma_{xx})^2 + 6(\sigma_{xy}^2+\sigma_{yz}^2+\sigma_{zx}^2)\big]}.
\end{equation}

For 2D, the expression reduces accordingly.

## Example Input File Syntax

!listing test/tests/mechanics/mech.i block=/TensorComputes/Postprocess/vonmises

!syntax parameters /TensorComputes/Solve/ComputeVonMisesStress

!syntax inputs /TensorComputes/Solve/ComputeVonMisesStress

!syntax children /TensorComputes/Solve/ComputeVonMisesStress

