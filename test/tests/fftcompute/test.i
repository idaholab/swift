[Mesh]
  type = FFTMesh
  dim = 2
  nx = 100
  ny = 100
  nz = 100
  xmax = ${fparse pi*4}
  ymax = ${fparse pi*4}
  zmax = ${fparse pi*4}
[]

[FFTBuffers]
  [eta]
  []
  [eta_bar]
  []
  [f]
  []
  [fbar]
  []
  [kappa_k2]
  []
[]

[TensorICs]
  [eta]
    type = SineIC
    buffer = eta
  []
  [kappa_k2]
    type = ReciprocalLaplacianFactor
    factor = 0.2
    buffer = kappa_k2
  []
[]

[FFTComputes]
  [f]
    type = ParsedCompute
    buffer = f
    enable_jit = true
    expression = '0.1*(eta+2)^2*(eta-2)^2'
    derivatives = eta
    inputs = eta
  []
  [fbar]
    type = PerformFFT
    buffer = fbar
    input = f
  []
  [eta_bar]
    type = PerformFFT
    buffer = eta_bar
    input = eta
  []
[]

[FFTTimeIntegrators]
  [eta]
    type = FFTSemiImplicit
    buffer = eta
    reciprocal_buffer = eta_bar
    linear_reciprocal = kappa_k2
    nonlinear_reciprocal = fbar
  []
[]

[AuxVariables]
  [eta]
  []
  [f]
  []
[]

[AuxKernels]
  [eta]
    type = FFTBufferAux
    buffer = eta
    variable = eta
    execute_on = TIMESTEP_END
  []
  [f]
    type = FFTBufferAux
    buffer = f
    variable = f
    execute_on = TIMESTEP_END
  []
[]

[Postprocessors]
  [min_eta]
    type = ElementExtremeValue
    variable = eta
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_eta]
    type = ElementExtremeValue
    variable = eta
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
  [F]
    type = ElementIntegralVariablePostprocessor
    variable = f
    execute_on = 'TIMESTEP_END'
  []
  [Eta]
    type = ElementIntegralVariablePostprocessor
    variable = eta
    execute_on = 'TIMESTEP_END'
  []
[]

[Problem]
  type = FFTProblem
[]

[Executioner]
  type = Transient
  num_steps = 100
  dt = 0.1
[]

[Outputs]
  exodus = true
  csv = true
  perf_graph = true
  execute_on = 'TIMESTEP_END'
[]
