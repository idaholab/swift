[Mesh]
  type = FFTMesh
  dim = 3
  nx = 100
  ny = 100
  nz = 100
  xmax = ${fparse pi*4}
  ymax = ${fparse pi*4}
  zmax = ${fparse pi*4}
  dummy_mesh = true
[]

[FFTBuffers]
  # phase field
  [c]
  []
  [cbar]
  []
  [mu]
  []
  [mubar]
  []
  [Mbarmubar]
  []

  # mechanics
  [disp_x]
  []
  [disp_y]
  []
  [disp_z]
  []

  # constant tensors
  [Mbar]
  []
  [kappabarbar]
  []
[]

[FFTOutputs]
  [xdmf]
    type = FFTRawXDMFOut
    buffer = 'c disp_x disp_y disp_z'
    output_mode = 'Node Node Node Node'
    enable_hdf5 = true
  []
[]

[FFTICs]
  [c]
    type = RandomFFTIC
    buffer = c
    min = 0.44
    max = 0.56
  []
  [disp_x]
    type = RandomFFTIC
    buffer = disp_x
    min = 0
    max = 0
  []
  [disp_y]
    type = RandomFFTIC
    buffer = disp_y
    min = 0
    max = 0
  []
  [disp_z]
    type = RandomFFTIC
    buffer = disp_z
    min = 0
    max = 0
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
    expression = '0.1*c^2*(c-1)^2 + c*sin(x/2)*0.005'
    extra_symbols = true
    derivatives = c
    # expression = "0.4*c^3-0.6*c^2+0.2*c"
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

  [qsmech]
    type = FFTQuasistaticElasticity
    displacements = 'disp_x disp_y disp_z'
    cbar = cbar
    lambda = 12
    mu = 4
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

[Postprocessors]
  [min_c]
    type = FFTExtremeValuePostprocessor
    buffer = c
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_c]
    type = FFTExtremeValuePostprocessor
    buffer = c
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []

  [min_disp_x]
    type = FFTExtremeValuePostprocessor
    buffer = disp_x
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_disp_x]
    type = FFTExtremeValuePostprocessor
    buffer = disp_x
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
  [min_disp_y]
    type = FFTExtremeValuePostprocessor
    buffer = disp_y
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_disp_y]
    type = FFTExtremeValuePostprocessor
    buffer = disp_y
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
  [min_disp_z]
    type = FFTExtremeValuePostprocessor
    buffer = disp_z
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_disp_z]
    type = FFTExtremeValuePostprocessor
    buffer = disp_z
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []

  [C]
    type = FFTIntegralPostprocessor
    buffer = c
    execute_on = 'TIMESTEP_END'
  []
  [cavg]
    type = FFTAveragePostprocessor
    buffer = c
    execute_on = 'TIMESTEP_END'
  []
[]

[Problem]
  type = FFTProblem
  spectral_solve_substeps = 1000
[]

[Executioner]
  type = Transient
  num_steps = 50
  [TimeStepper]
    type = IterationAdaptiveDT
    growth_factor = 1.8
    dt = 0.1
  []
  dtmax = 500
[]

[Outputs]
  csv = true
  perf_graph = true
  execute_on = 'TIMESTEP_END'
[]
