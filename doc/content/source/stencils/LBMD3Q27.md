# LBMD3Q27

!syntax description /Stencil/LBMD3Q27

LBMD3Q27 objects creatres D3Q27 stencil. This object holds tensors for directions and weights of streaming, momentum and inverse transformation matrices, diagonal relaxation matrix, and the indices of boundaries required for boundary conditions.

## Overview

Constructs a 3\-D, 27\-velocity LBM stencil (D3Q27) and exposes its discrete velocities, weights,
and indexing used by streaming and boundary conditions.

## Example Input File Syntax

!listing examples/lbm/Formula1-aerodynamics/f1.i block=/Stencil

!syntax parameters /Stencil/LBMD3Q27

!syntax inputs /Stencil/LBMD3Q27

!syntax children /Stencil/LBMD3Q27
