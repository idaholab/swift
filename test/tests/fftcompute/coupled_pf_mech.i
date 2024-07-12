[Mesh]
  type = FFTMesh
  dim = 3
  nx = 128
  ny = 128
  nz = 128
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
  [mumechbar]
  []
  [mumech]
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
    buffer = 'c disp_x disp_y disp_z mu mumech'
    output_mode = 'Node Node Node Node Cell Cell'
    enable_hdf5 = true
  []
[]

[TensorICs]
  [c]
    type = RandomTensorIC
    buffer = c
    min = 0.44
    max = 0.56
  []
  [disp_x]
    type = RandomTensorIC
    buffer = disp_x
    min = 0
    max = 0
  []
  [disp_y]
    type = RandomTensorIC
    buffer = disp_y
    min = 0
    max = 0
  []
  [disp_z]
    type = RandomTensorIC
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
    # chemical potential (real space)
    type = ParsedCompute
    buffer = mu
    enable_jit = true
    expression = '0.1*c^2*(c-1)^2' # + c*sin(x/2)*0.005'
    extra_symbols = true
    derivatives = c
    inputs = c
  []
  [mubar]
    # chemical potential (reciprocal space)
    type = PerformFFT
    buffer = mubar
    input = mu
  []
  [mumechbar]
    # mechanical chemical potential (reciprocal space)
    type = FFTElasticChemicalPotential
    buffer = mumechbar
    cbar = cbar
    displacements = 'disp_x disp_y disp_z'
    lambda = 100
    mu = 50
    e0 = 0.02
  []
  [mumech]
    # chemical potential (reciprocal space)
    type = PerformFFT
    forward = false
    buffer = mumech
    input = mumechbar
  []

  [Mbarmubar]
    type = ParsedCompute
    buffer = Mbarmubar
    enable_jit = true
    expression = 'Mbar*(mubar+mumechbar)'
    inputs = 'Mbar mubar mumechbar'
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
    lambda = 100
    mu = 50
    e0 = 0.02
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
  print_debug_output = true
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
  csv = true
  perf_graph = true
  execute_on = 'TIMESTEP_END'
[]
