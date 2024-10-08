[Domain]
  dim = 2
  nx = 50
  ny = 50
  xmax = ${fparse pi*4}
  ymax = ${fparse pi*4}

  device_names = 'cuda'

  mesh_mode = DOMAIN
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

[TensorComputes]
  [Initialize]
    [eta]
      type = ParsedTensor
      buffer = eta
      function = 'sin(x)+sin(y)+sin(z)'
    []
    [zero]
      type = ConstantTensor
      buffer = zero
      real = 0
    []
  []

  [Solve]
    [eta_bar]
      type = ForwardFFT
      buffer = eta_bar
      input = eta
    []
    [eta_2]
      type = InverseFFT
      buffer = eta2
      input = eta_bar
    []
  []
[]

[TensorTimeIntegrators]
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
