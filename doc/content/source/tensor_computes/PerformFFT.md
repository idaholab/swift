# ForwardFFT / InverseFFT

!syntax description /TensorComputes/Solve/ForwardFFT

!syntax description /TensorComputes/Solve/InverseFFT

Perform fast Fourier transforms on tensor buffers. `ForwardFFT` maps a real-space buffer to its
reciprocal representation; `InverseFFT` maps the reciprocal buffer back to real space.

## Example Input File Syntax

!listing
[TensorComputes]
  [Solve]
    [to_k]
      type = ForwardFFT
      buffer = cbar
      input = c
    []
    [back_to_x]
      type = InverseFFT
      buffer = c
      input = cbar
    []
  []
[]
!listing-end

!syntax parameters /TensorComputes/Solve/ForwardFFT

!syntax inputs /TensorComputes/Solve/ForwardFFT

!syntax children /TensorComputes/Solve/ForwardFFT

!syntax parameters /TensorComputes/Solve/InverseFFT

!syntax inputs /TensorComputes/Solve/InverseFFT

!syntax children /TensorComputes/Solve/InverseFFT
