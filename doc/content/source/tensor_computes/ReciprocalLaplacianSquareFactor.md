# ReciprocalLaplacianSquareFactor

!syntax description /TensorComputes/Solve/ReciprocalLaplacianSquareFactor

Fills a tensor with

\begin{equation}
\mathbf{k}^4\cdot f
\end{equation}

where $\mathbf{k}$ is the k-vector. This is the Fourier transform of the squared Laplacian operator.

## Example Input File Syntax

!listing test/tests/cahnhilliard/cahnhilliard.i block=TensorComputes/Initialize/kappabarbar

!syntax parameters /TensorComputes/Solve/ReciprocalLaplacianSquareFactor

!syntax inputs /TensorComputes/Solve/ReciprocalLaplacianSquareFactor

!syntax children /TensorComputes/Solve/ReciprocalLaplacianSquareFactor
