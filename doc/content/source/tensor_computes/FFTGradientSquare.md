# FFTGradientSquare

!syntax description /TensorComputes/Solve/FFTGradientSquare

Compute the Laplacian (gradient squared) of a real space tensor via reciprocal space
spectral gradient. Internally this involves a forward and a backward Fourier transform
and invetwee a multiplication with $-k^2$.

## Overview

Computes `|\nabla u|^2` using spectral derivatives: forward FFT, multiply by wave vectors, and
sum component squares. Provide the source field via
[!param](/TensorComputes/Solve/FFTGradientSquare/input) and the destination via
[!param](/TensorComputes/Solve/FFTGradientSquare/buffer).

## Example Input File Syntax

!listing test/tests/gradient/gradient_square.i block=TensorComputes/Initialize/grad_sq

!syntax parameters /TensorComputes/Solve/FFTGradientSquare

!syntax inputs /TensorComputes/Solve/FFTGradientSquare

!syntax children /TensorComputes/Solve/FFTGradientSquare
