# ComputeDisplacements

!syntax description /TensorComputes/Solve/FFTElasticChemicalPotential

Integrates a deformation gradient tensor $\mathbf{F}$ to build a displacement vector. The compute first calculates the average deformation gradient tensor $\mathbf{F_{box}}$, which encodes the affine diaplacement transformation $\vec u_{aff} = (\mathbf{F_{box}} - \mathbf{I}) \cdot \vec x$.
Then it subtracts the average to obtain the periodic perturtbation $\mathbf{F_{per}} = \mathbf{F}-\mathbf{F_{box}}$, and its Fourier transform $\hat{\mathbf{H}} = \mathcal F(\mathbf{F_{per}})$.

\begin{equation}
\hat u_{per,i} = \sum_j \frac{i\vec k_j \cdot \hat H_{ij}}{\vec k^2}
\end{equation}

With $\vec u_{per} = {\mathcal F}^{-1}(\mathbf{\hat{\vec u_{per}}})$ the displacement is calculated as

\begin{equation}
\vec u = \vec u_{aff} + \vec u_{per}
\end{equation}

!syntax parameters /TensorComputes/Postprocess/ComputeDisplacements

!syntax inputs /TensorComputes/Postprocess/ComputeDisplacements

!syntax children /TensorComputes/Postprocess/ComputeDisplacements


## Visualizing displaced domains in Paraview

1. When using [XDMF output](XDMFTensorOutput.md), ensure that the [!param](/TensorOutputs/XDMFTensorOutput/transpose) parameter is set to `true` (default). This counters the Paraview XDMF reader [issue described here](https://discourse.paraview.org/t/axis-swapped-with-xdmf-topologytype-3dcorectmesh/3059/4).

2. Add a `Calculator` filter with
- `iHat * disp_x + jHat * disp_y + kHat * disp_z` for 3d
- `jHat * disp_x + kHat * disp_y` for 2d (this is the part of the +cooked+ XDMF implememntation in Paraview that the `transpose` can not completely fix)

3. Add a `Warp By Vector` filter to apply the result vector from the calculator as displacements

