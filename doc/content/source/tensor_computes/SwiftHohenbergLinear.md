# SwiftHohenbergLinear

!syntax description /TensorComputes/Solve/SwiftHohenbergLinear

Fills a tensor with

\begin{equation}
r - \alpha^2 (1 - \mathbf{k}^2)^2;
\end{equation}

where $\mathbf{k}$ is the k-vector. This is the Fourier transform of the linear
term in the Swift-Hohenberg equation.

## Example Input File Syntax

!listing test/tests/tensor_compute/rotating_grain_secant.i block=TensorComputes/Initialize/linear

!syntax parameters /TensorComputes/Solve/SwiftHohenbergLinear

!syntax inputs /TensorComputes/Solve/SwiftHohenbergLinear

!syntax children /TensorComputes/Solve/SwiftHohenbergLinear
