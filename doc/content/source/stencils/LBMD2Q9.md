# LBMD2Q9

!syntax description /Stencil/LBMD2Q9

LBMD2Q9 objects creatres D2Q9 stencil. This object holds tensors for directions and weights of streaming, momentum and inverse transformation matrices, diagonal relaxation matrix, and the indices of boundaries required for boundary conditions.

## Overview

Constructs a 2\-D, 9\-velocity LBM stencil (D2Q9) and exposes its discrete velocities, weights,
and indexing used by streaming and boundary conditions.

## Example Input File Syntax

!listing test/tests/lbm/channel2D.i block=Stencil/d2q9

!syntax parameters /Stencil/LBMD2Q9

!syntax inputs /Stencil/LBMD2Q9

!syntax children /Stencil/LBMD2Q9
