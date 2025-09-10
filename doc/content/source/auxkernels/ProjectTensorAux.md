# ProjectTensorAux

!syntax description /AuxKernels/ProjectTensorAux

This AuxKernel works on nodal and elemental variables. For elemental variables the element
centroid location is uses to look up the corresponsing co-located tensor entry. For nodal
variables a shift of half a grid spacing is applied to the nodal coordinates before looking
up the co-located tensor entry (wraping around periodically).

The ideal mesh in either case is a generated mesh with the same dimensions as the FFT domain
and the same number of elements as grid cells in each dimension.

Using the [Domain action](/DomainAction.md) such a mesh can be set up automatically.

## Overview

Projects a tensor buffer to a mesh variable by sampling the buffer at cell centroids (elemental
variables) or at nodal locations shifted by half a grid spacing with periodic wrap (nodal
variables). Select the source tensor via [!param](/AuxKernels/ProjectTensorAux/buffer) and the
target with [!param](/AuxKernels/ProjectTensorAux/variable).

## Example Input File Syntax

!listing test/tests/cahnhilliard/cahnhilliard.i block=AuxKernels/c

!syntax parameters /AuxKernels/ProjectTensorAux

!syntax inputs /AuxKernels/ProjectTensorAux

!syntax children /AuxKernels/ProjectTensorAux
