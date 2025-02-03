# XDMFTensorOutput

!syntax description /TensorOutputs/XDMFTensorOutput

This TensorOutput object writes tensors to an [XDMF](https://www.xdmf.org/) data file set.
The parameter [!param](/TensorOutput/XDMFTensorOutput/enable_hdf5) enables writing of the raw
tensor data to an HDF5 file. The resulting data items will be bamed `/var.n`, where `var` is the name of the
exported tensor and `n` is the simulation step. Output as cell or node centered data is supported
and can be selected using the [!param](/TensorOutput/XDMFTensorOutput/output_mode) parameter. Cell centered output
results in a value per simulation grid cell (e.g. `N[0] * N[1] * N[2]` entries), while for node centered output
the cell edge nodes are periodically replicated, resulting in `(N[0]+1) * (N[1]+1) * (N[2]+1)` exported entries.

## Overview

!! Replace these lines with information regarding the XDMFTensorOutput object.

## Example Input File Syntax

!! Describe and include an example of how to use the XDMFTensorOutput object.

!syntax parameters /TensorOutputs/XDMFTensorOutput

!syntax inputs /TensorOutputs/XDMFTensorOutput

!syntax children /TensorOutputs/XDMFTensorOutput
