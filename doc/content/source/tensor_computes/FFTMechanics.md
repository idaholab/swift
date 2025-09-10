# FFTMechanics

!syntax description /TensorComputes/Solve/FFTMechanics

Solve a heterogeneous mechanics problem using the approach by deGeus et al.  [!cite](DEGEUS2017412).

## Example Input File Syntax

!listing
[TensorComputes]
  [Solve]
    [root]
      [applied_strain]
        type = MacroscopicShearTensor
        buffer = applied_strain
        F = F
      []
      [hyper_elasticity]
        type = HyperElasticIsotropic
        buffer = stress
        F = Fnew
        K = K
        mu = mu
      []
      [mech]
        type = FFTMechanics
        buffer = Fnew
        F = F
        K = K
        mu = mu
        l_tol = 1e-2
        nl_rel_tol = 2e-2
        nl_abs_tol = 2e-2
        constitutive_model = hyper_elasticity
        stress = stress
        tangent_operator = dstressdstrain
        applied_macroscopic_strain = applied_strain
      []
    []
  []
[]
!listing-end

Notes:
- Supplies and updates the deformation gradient buffer `Fnew`.
- Requires a constitutive-model compute that provides `stress` and its tangent.
- An optional macroscopic strain buffer lets you impose a prescribed average deformation.

See also: [MacroscopicShearTensor](MacroscopicShearTensor.md),
[HyperElasticIsotropic](HyperElasticIsotropic.md),
[ComputeDisplacements](ComputeDisplacements.md).

!syntax parameters /TensorComputes/Solve/FFTMechanics

!syntax inputs /TensorComputes/Solve/FFTMechanics

!syntax children /TensorComputes/Solve/FFTMechanics

!bibtex bibliography
