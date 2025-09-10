# ConstantTensor

!syntax description /TensorComputes/Solve/ConstantTensor

`ConstantTensor` and `ConstantReciprocalTensor` set a tensor to a constant value in real space and reciprocal space respectively. The reciprocal space version takes an [!param](/TensorComputes/Solve/ConstantReciprocalTensor/imaginary) value parameter.

## Overview

Writes a constant scalar into the target buffer each time it runs. Use
[!param](/TensorComputes/Solve/ConstantTensor/buffer) to select the destination and
[!param](/TensorComputes/Solve/ConstantTensor/real) to provide the value.

## Example Input File Syntax

!listing test/tests/neml2/scalar.i block=TensorComputes/Initialize/A

!syntax parameters /TensorComputes/Solve/ConstantTensor

!syntax inputs /TensorComputes/Solve/ConstantTensor

!syntax children /TensorComputes/Solve/ConstantTensor
