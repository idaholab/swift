# PlainTensorBuffer

!syntax description /TensorBuffers/PlainTensorBuffer

A `PlainTensorBuffer` holds a plain vanilla `torch::Tensor` and is interpreted as a buffer of scalar values.

## Overview

Generic N\-dimensional tensor buffer for scalar, vector, or user\-defined data not tied to an LBM
stencil. Optional [!param](/TensorBuffers/PlainTensorBuffer/map_to_aux_variable) enables fast projection onto a
matching mesh variable.

## Example Input File Syntax

!listing test/tests/gradient/gradient_square.i block=TensorBuffers/grad_sq

!syntax parameters /TensorBuffers/PlainTensorBuffer

!syntax inputs /TensorBuffers/PlainTensorBuffer

!syntax children /TensorBuffers/PlainTensorBuffer
