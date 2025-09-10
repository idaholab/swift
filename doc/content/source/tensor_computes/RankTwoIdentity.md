# RankTwoIdentity

!syntax description /TensorComputes/Solve/RankTwoIdentity

RankTwoIdentity fills the output with the rank-2 identity tensor in real space, expanded to the full buffer shape.

## Overview

For a problem dimension `d`, the object forms `I_d` and broadcasts it over the spatial grid, producing a field with shape `(..., d, d)`. This is useful as a building block for constitutive models and tests.

## Example Input File Syntax

!listing
[
  TensorComputes/
    [./I]
      type = RankTwoIdentity
      buffer = identity
    [../]
]
!listing-end

!syntax parameters /TensorComputes/Solve/RankTwoIdentity

!syntax inputs /TensorComputes/Solve/RankTwoIdentity

!syntax children /TensorComputes/Solve/RankTwoIdentity

