# SecantSolver

!alert construction title=Undocumented Class
The SecantSolver has not been documented. The content listed below should be used as a starting point for
documenting the class, which includes the typical automatic documentation associated with a
MooseObject; however, what is contained is ultimately determined by what is necessary to make the
documentation clear for users.

!syntax description /TensorSolver/SecantSolver

## Overview

Nonlinear solver using a secant update of the reciprocal nonlinear operator to accelerate fixed\-point
iterations. Useful when a Jacobian is expensive or unavailable.

## Example Input File Syntax

!listing test/tests/tensor_compute/rotating_grain_secant.i block=TensorSolver

!syntax parameters /TensorSolver/SecantSolver

!syntax inputs /TensorSolver/SecantSolver

!syntax children /TensorSolver/SecantSolver
