# DeAliasingTensor

!syntax description /TensorComputes/Solve/DeAliasingTensor

Creates a spectral de-aliasing filter in reciprocal space. Two methods are
provided:

- SHARP: a two-thirds truncation ("2/3 rule") that zeros high-frequency modes.
  The filter value \(\sigma(\mathbf{k})\) is

\begin{equation}
\sigma(\mathbf{k}) = \begin{cases}
0, & \text{if } |k_x| \gt \tfrac{2}{3} k_{x,\max} \text{ or } |k_y| \gt \tfrac{2}{3} k_{y,\max} \text{ or } |k_z| \gt \tfrac{2}{3} k_{z,\max}, \\
1, & \text{otherwise.}
\end{cases}
\end{equation}

- HOULI: the Hou-Li exponential filter, which smoothly damps high-frequency
  content using

\begin{equation}
\sigma(\mathbf{k}) = \exp\!\left(-\alpha\,\left(|k_x|^{p} + |k_y|^{p} + |k_z|^{p}\right)\right),
\end{equation}

where \(p\) is the exponent and \(\alpha\) is the pre-factor controlling the
sharpness of the damping. The resulting filter \(\sigma\) is applied element-wise
in reciprocal space.

## Example Input File Syntax

!syntax parameters /TensorComputes/Solve/DeAliasingTensor

!syntax inputs /TensorComputes/Solve/DeAliasingTensor

!syntax children /TensorComputes/Solve/DeAliasingTensor

