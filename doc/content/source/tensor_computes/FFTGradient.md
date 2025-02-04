# FFTGradient

!syntax description /TensorComputes/Solve/FFTGradient

Compute the gradient component of a real space tensor along a selected axis using the
spectral gradient. Internally this involves a forward and a backward Fourier transform
and inbetween a multiplication with the $\vec k$ vector and the imaginary unit $i$.

## Example Input File Syntax

!! Describe and include an example of how to use the FFTGradient object.

!syntax parameters /TensorComputes/Solve/FFTGradient

!syntax inputs /TensorComputes/Solve/FFTGradient

!syntax children /TensorComputes/Solve/FFTGradient
