# MooseFunctionTensor

!syntax description /TensorComputes/Solve/MooseFunctionTensor

## Overview

Evaluates a MOOSE `Function` on the domain and writes the values into the target buffer. Select
the output with [!param](/TensorComputes/Solve/MooseFunctionTensor/buffer) and provide the function
object name via [!param](/TensorComputes/Solve/MooseFunctionTensor/function).

## Example Input File Syntax

!listing test/tests/tensor_compute/rotating_grain_secant.i block=TensorComputes/Initialize/psi

!syntax parameters /TensorComputes/Solve/MooseFunctionTensor

!syntax inputs /TensorComputes/Solve/MooseFunctionTensor

!syntax children /TensorComputes/Solve/MooseFunctionTensor
