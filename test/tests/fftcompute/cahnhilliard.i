[Mesh]
  type = FFTMesh
  dim = 3
  nx = 200
  ny = 200
  nz = 200
  xmax = ${fparse pi*8}
  ymax = ${fparse pi*8}
  zmax = ${fparse pi*8}
[]

[FFTBuffers]
  [c]
    map_to_aux_variable = c
  []
  [cbar]
  []
  [mu]
    map_to_aux_variable = mu
  []
  [mubar]
  []
  [Mbarmubar]
  []
  # constant tensors
  [Mbar]
  []
  [kappabarbar]
  []
[]

[TensorICs]
  [c]
    type = RandomFFTIC
    buffer = c
    min = 0.44
    max = 0.56
  []
  [Mbar]
    type = ReciprocalLaplacianFactor
    factor = 0.2 # Mobility
    buffer = Mbar
  []
  [kappabarbar]
    type = ReciprocalLaplacianSquareFactor
    factor = -0.001 # kappa
    buffer = kappabarbar
  []
[]

[FFTComputes]
  [mu]
    type = ParsedCompute
    buffer = mu
    enable_jit = true
    # expression = '0.1*c^2*(c-1)^2'
    # derivatives = c
    expression = "0.4*c^3-0.6*c^2+0.2*c"
    inputs = c
  []
  [mubar]
    type = PerformFFT
    buffer = mubar
    input = mu
  []
  [Mbarmubar]
    type = ParsedCompute
    buffer = Mbarmubar
    enable_jit = true
    expression = 'Mbar*mubar'
    inputs = 'Mbar mubar'
  []
  [cbar]
    type = PerformFFT
    buffer = cbar
    input = c
  []
[]

[FFTTimeIntegrators]
  [c]
    type = FFTSemiImplicit
    buffer = c
    reciprocal_buffer = cbar
    linear_reciprocal = kappabarbar
    nonlinear_reciprocal = Mbarmubar
  []
[]

[AuxVariables]
  [mu]
    family = MONOMIAL
    order = CONSTANT
  []
  [c]
    # family = MONOMIAL
    # order = CONSTANT
  []
[]

[AuxKernels]
  # [c]
  #   type = FFTBufferAux
  #   buffer = c
  #   variable = c
  #   execute_on = final
  # []
  # [f]
  #   type = FFTBufferAux
  #   buffer = f
  #   variable = f
  #   execute_on = TIMESTEP_END
  # []
[]

[Postprocessors]
  [min_c]
    type = ElementExtremeValue
    variable = c
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_c]
    type = ElementExtremeValue
    variable = c
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
  # [F]
  #   type = ElementIntegralVariablePostprocessor
  #   variable = f
  #   execute_on = 'TIMESTEP_END'
  # []
  [C]
    type = ElementIntegralVariablePostprocessor
    variable = c
    execute_on = 'TIMESTEP_END'
  []
[]

[Problem]
  type = FFTProblem
  spectral_solve_substeps = 1000
[]

[Executioner]
  type = Transient
  num_steps = 100
  [TimeStepper]
    type = IterationAdaptiveDT
    growth_factor = 1.8
    dt = 0.1
  []
  dtmax = 1000
[]

[Outputs]
  exodus = true
  csv = true
  perf_graph = true
  execute_on = 'TIMESTEP_END'
[]
