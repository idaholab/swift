# MacroscopicShearTensor

!syntax description /TensorComputes/Solve/MacroscopicShearTensor

Builds a macroscopic applied shear deformation gradient and subtracts the volume-averaged
deformation gradient $\langle F \rangle$ to form the fluctuation to be imposed in a mechanics solve.

Specifically, it starts from the identity tensor $I$ and adds a time-dependent shear in the
`(0,1)` component, $F_{01} \leftarrow F_{01} + t$, producing

\begin{equation}
F_\text{macro}(t) = I + t\, e_0 \otimes e_1.
\end{equation}

The output buffer is then

\begin{equation}
F_\text{applied}(t) = F_\text{macro}(t) - \langle F \rangle.
\end{equation}

## Example Input File Syntax

In this example, `MacroscopicShearTensor` provides the applied macroscopic strain used by
`FFTMechanics` during the solve.

!listing
[TensorComputes]
  [Solve]
    [root]
      [applied_strain]
        type = MacroscopicShearTensor
        buffer = applied_strain
        F = F
      []
      [mech]
        type = FFTMechanics
        buffer = Fnew
        F = F
        K = K
        mu = mu
        constitutive_model = hyper_elasticity
        stress = stress
        applied_macroscopic_strain = applied_strain
      []
    []
  []
[]
!listing-end

See also: [FFTMechanics](FFTMechanics.md), [ComputeDisplacements](ComputeDisplacements.md).

!syntax parameters /TensorComputes/Solve/MacroscopicShearTensor

!syntax inputs /TensorComputes/Solve/MacroscopicShearTensor

!syntax children /TensorComputes/Solve/MacroscopicShearTensor

