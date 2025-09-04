# ProjectVectorTensorAux

!syntax description /AuxKernels/ProjectVectorTensorAux

ProjectVectorTensorAux samples a vectorial TensorBuffer and projects its components onto an auxiliary array variable. This object is currently under development and compiled out in the source.

## Overview

Given a coupled vector buffer, the kernel evaluates the buffer at the node or element location and writes the selected components into the target aux variable entries.

## Example Input File Syntax

!listing
[
  AuxVariables/
    [./u_vec]
      family = MONOMIAL
      order  = CONSTANT
      components = '0 1 2'
    [../]
  AuxKernels/
    [./sample_u]
      type   = ProjectVectorTensorAux
      variable = u_vec
      buffer = velocity
    [../]
]
!listing-end

!syntax parameters /AuxKernels/ProjectVectorTensorAux

!syntax inputs /AuxKernels/ProjectVectorTensorAux

!syntax children /AuxKernels/ProjectVectorTensorAux

