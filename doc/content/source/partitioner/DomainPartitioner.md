# DomainPartitioner

!syntax description /Mesh/Partitioner/DomainPartitioner

DomainPartitioner partitions the mesh by overlaying a uniform grid and assigning all elements that fall into the same grid cell to the same processor. The number of grid cells per direction is user-controllable.

## Overview

Parameters `nx`, `ny`, `nz` set the number of partitions along x, y, z. If their product does not match the number of partitions, a balanced 1d/2d/3d factorization is computed automatically. This partitioner is helpful for problems with structured spatial locality.

## Example Input File Syntax

!listing
[
  Mesh/
    [./mesh]
      type = GeneratedMesh
      dim = 3
      nx = 64
      ny = 64
      nz = 64
      partitioner = DomainPartitioner
      partitioner_options = 'nx=4 ny=2 nz=2'
    [../]
]
!listing-end

!syntax parameters /Mesh/Partitioner/DomainPartitioner

!syntax inputs /Mesh/Partitioner/DomainPartitioner

!syntax children /Mesh/Partitioner/DomainPartitioner

