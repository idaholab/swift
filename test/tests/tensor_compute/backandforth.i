[Mesh]
  type = UniformTensorMesh
  dim = 2
  nx = 50
  ny = 50
  xmax = ${fparse pi*4}
  ymax = ${fparse pi*4}
[]

[TensorBuffers]
  [eta]
  []
  [eta_bar]
  []
  [eta2]
  []
  [zero]
  []
[]

[TensorICs]
  [eta]
    type = SineIC
    buffer = eta
  []
  [zero]
    type = ConstantTensorIC
    buffer = zero
    real = 0
  []
[]

[FFTComputes]
  [eta_bar]
    type = PerformFFT
    buffer = eta_bar
    input = eta
  []
  [eta_2]
    type = PerformFFT
    buffer = eta2
    input = eta_bar
    forward = false
  []
[]

[FFTTimeIntegrators]
  [eta]
    type = FFTSemiImplicit
    buffer = eta
    reciprocal_buffer = eta_bar
    linear_reciprocal = zero
    nonlinear_reciprocal = zero
  []
[]

[AuxVariables]
  [eta]
  []
[]

[AuxKernels]
  [eta]
    type = ProjectTensorAux
    buffer = eta
    variable = eta
    execute_on = TIMESTEP_END
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 4
[]

[Outputs]
  exodus = true
[]
