[Domain]
  dim = 3
  nx = 128
  ny = 128
  nz = 128
  xmax = ${fparse pi*4}
  ymax = ${fparse pi*4}
  zmax = ${fparse pi*4}

  device_names = 'cuda:3'

  mesh_mode = DUMMY
[]

[TensorBuffers]
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

[TensorOutputs]
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'c disp_x disp_y disp_z mu mumech'
    output_mode = 'Node Node Node Node Cell Cell'
    enable_hdf5 = true
  []
[]

[TensorComputes]
  [Initialize]
    [c]
      type = RandomTensor
      buffer = c
      min = 0.44
      max = 0.56
    []
    [mu_init]
      type = ConstantTensor
      buffer = mu
    []
    [mumech_init]
      type = ConstantTensor
      buffer = mumech
    []
    [disp_x]
      type = RandomTensor
      buffer = disp_x
      min = 0
      max = 0
    []
    [disp_y]
      type = RandomTensor
      buffer = disp_y
      min = 0
      max = 0
    []
    [disp_z]
      type = RandomTensor
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

  [Solve]
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
      type = ForwardFFT
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
      type = InverseFFT
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
      type = ForwardFFT
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
[]

[TensorSolver]
  type = SecantSolver
  substeps = 1
  max_iterations = 1000
  # damping = 0.75
  relative_tolerance = 1e-6
  absolute_tolerance = 1e-6
  buffer = c
  dt_epsilon = 1e-7
  reciprocal_buffer = cbar
  linear_reciprocal = kappabarbar
  nonlinear_reciprocal = Mbarmubar
  verbose = true
[]

[Postprocessors]
  [min_c]
    type = TensorExtremeValuePostprocessor
    buffer = c
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_c]
    type = TensorExtremeValuePostprocessor
    buffer = c
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []

  [min_disp_x]
    type = TensorExtremeValuePostprocessor
    buffer = disp_x
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_disp_x]
    type = TensorExtremeValuePostprocessor
    buffer = disp_x
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
  [min_disp_y]
    type = TensorExtremeValuePostprocessor
    buffer = disp_y
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_disp_y]
    type = TensorExtremeValuePostprocessor
    buffer = disp_y
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
  [min_disp_z]
    type = TensorExtremeValuePostprocessor
    buffer = disp_z
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_disp_z]
    type = TensorExtremeValuePostprocessor
    buffer = disp_z
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []

  [C]
    type = TensorIntegralPostprocessor
    buffer = c
    execute_on = 'TIMESTEP_END'
  []
  [cavg]
    type = TensorAveragePostprocessor
    buffer = c
    execute_on = 'TIMESTEP_END'
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  end_time = 100
  [TimeStepper]
    type = TensorSolveIterationAdaptiveDT
    dt = 0.1
    max_iterations = 500
    min_iterations = 300
    growth_factor = 1.1
    cutback_factor = 0.9
  []
[]

[Outputs]
  csv = true
  perf_graph = true
  execute_on = 'TIMESTEP_END'
[]