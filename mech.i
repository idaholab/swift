[Domain]
  dim = 3
  nx = 30
  ny = 32
  nz = 34
  xmax = ${fparse 2*pi}
  ymax = ${fparse 2*pi}
  zmax = ${fparse 2*pi}
  mesh_mode = DUMMY
[]

[Problem]
  type = TensorProblem
[]

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

    [mech]
      type = FFTMechanics
      buffer = F
      K = K
      mu = mu
    []
  []
  [Postprocess]
    [displacements]
      type = ComputeDisplacements
      buffer = disp
      F = F
    []
    [vonmises]
      type = ComputeVonMisesStress
      buffer = sV
    []
  []
[]

[TensorOutputs]
  [deformation_tensor]
    type = XDMFTensorOutput
    buffer = 'disp sV F'
    output_mode = 'OVERSIZED_NODAL CELL CELL'
    enable_hdf5 = true
  []
[]

[Executioner]
  type = Transient
  num_steps = 1
[]
