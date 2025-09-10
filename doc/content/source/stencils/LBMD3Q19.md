# LBMD3Q19

!syntax description /Stencil/LBMD3Q19

LBMD3Q19 objects creatres D3Q19 stencil. This object holds tensors for directions and weights of streaming, momentum and inverse transformation matrices, diagonal relaxation matrix, and the indices of boundaries required for boundary conditions.

## Overview

Constructs a 3\-D, 19\-velocity LBM stencil (D3Q19) and exposes its discrete velocities, weights,
and indexing used by streaming and boundary conditions.

## Example Input File Syntax

!listing test/tests/lbm/neumann_box.i block=Stencil/d3q19

!syntax parameters /Stencil/LBMD3Q19

!syntax inputs /Stencil/LBMD3Q19

!syntax children /Stencil/LBMD3Q19
