# BroydenSolver

!alert construction title=Undocumented Class
The BroydenSolver has not been documented. The content listed below should be used as a starting point for
documenting the class, which includes the typical automatic documentation associated with a
MooseObject; however, what is contained is ultimately determined by what is necessary to make the
documentation clear for users.

!syntax description /TensorSolver/BroydenSolver

## Overview

Quasi\-Newton nonlinear solver that updates an approximate Jacobian using Broyden's method to
accelerate fixed\-point iterations on the nonlinear reciprocal term.

## Example Input File Syntax

!listing benchmarks/02_oswald_ripening/2a_broyden.i block=TensorSolver

!syntax parameters /TensorSolver/BroydenSolver

!syntax inputs /TensorSolver/BroydenSolver

!syntax children /TensorSolver/BroydenSolver
