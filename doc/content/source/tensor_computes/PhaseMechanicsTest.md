# PhaseMechanicsTest

!syntax description /TensorComputes/Initialize/PhaseMechanicsTest

Creates a binary phase indicator tensor used in DeGeus-style mechanics tests. The tensor is
initialized to zero everywhere, and a rectangular sub-region is set to `1.0` to represent the
second phase. The selected sub-region depends on problem dimension:

- 3D: a block at the high end in `x` and `z`, and the low end in `y` (size `s = 9`).
- 2D: a block at the high end in `x` and the low end in `y` (size `s = 30`).

## Example Input File Syntax

Below, `PhaseMechanicsTest` seeds the phase field; elastic moduli are then mixed from the phase.

!listing
[TensorComputes]
  [Initialize]
    [phase]
      type = PhaseMechanicsTest
      buffer = phase
    []
    [K]
      type = ParsedCompute
      buffer = K
      expression = '(1-phase)*Ka + phase*Kb'
      inputs = phase
      constant_names = 'Ka Kb'
      constant_expressions = '0.833 8.33'
    []
    [mu]
      type = ParsedCompute
      buffer = mu
      expression = '(1-phase)*mua + phase*mub'
      inputs = phase
      constant_names = 'mua mub'
      constant_expressions = '0.386 3.86'
    []
  []
[]
!listing-end

See also: [HyperElasticIsotropic](HyperElasticIsotropic.md),
[FFTMechanics](FFTMechanics.md).

!syntax parameters /TensorComputes/Initialize/PhaseMechanicsTest

!syntax inputs /TensorComputes/Initialize/PhaseMechanicsTest

!syntax children /TensorComputes/Initialize/PhaseMechanicsTest

