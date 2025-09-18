# ReciprocalLaplacianFactor

!syntax description /TensorComputes/Solve/ReciprocalLaplacianFactor

Fills a tensor with

\begin{equation}
-\mathbf{k}^2\cdot f
\end{equation}

where $\mathbf{k}$ is the k-vector. This is the Fourier transform of the Laplacian operator.

## Example Input File Syntax

!listing benchmarks/01_spinodal_decomposition/1a.i block=TensorComputes/Initialize/kappabarbar

!syntax parameters /TensorComputes/Solve/ReciprocalLaplacianFactor

!syntax inputs /TensorComputes/Solve/ReciprocalLaplacianFactor

!syntax children /TensorComputes/Solve/ReciprocalLaplacianFactor
