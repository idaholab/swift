[Mesh]
  type = UniformTensorMesh
  dim = 3
  nx = 100
  ny = 100
  nz = 100
  xmax = ${fparse pi*4}
  ymax = ${fparse pi*4}
  zmax = ${fparse pi*4}
  dummy_mesh = true
[]

[TensorBuffers]
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
  # constant tensors
  [Mbar]
  []
  [kappabarbar]
  []
[]

[FFTOutputs]
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'c mu'
    output_mode = 'Node Cell'
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
    expression = '0.1*c^2*(c-1)^2'
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
  spectral_solve_substeps = 1000
[]

[Executioner]
  type = Transient
  num_steps = 20
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
