#
# Cahn-Hilliard solve using an explicit stabilized ROCK2 (RKC) integrator.
# This mirrors the semi-implicit ABM example but forms the explicit RHS in
# reciprocal space and advances with fixed-stage ROCK2.
#

[Domain]
  dim = 2
  nx = 200
  ny = 200

  xmax = ${fparse pi*8}
  ymax = ${fparse pi*8}

  # automatically create a matching mesh
  mesh_mode = DOMAIN
[]

[TensorBuffers]
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
  [rhsbar]
  []
  # constant tensors
  [Mbar]
  []
  [kappabarbar]
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

    # precompute fixed factors
    [Mbar]
      type = ReciprocalLaplacianFactor
      factor = 0.2 # Mobility M (Mbar = -M k^2)
      buffer = Mbar
    []
    [kappabarbar]
      type = ReciprocalLaplacianSquareFactor
      factor = -0.001 # -M*κ so that RHS has -M κ k^4 cbar
      buffer = kappabarbar
    []
    [mu_init]
      type = ConstantTensor
      buffer = mu
      real = 0
    []
  []

  [Solve]
    [mu]
      type = ParsedCompute
      buffer = mu
      enable_jit = true
      expression = '0.1*c^2*(c-1)^2'
      derivatives = c
      inputs = c
    []
    [mubar]
      type = ForwardFFT
      buffer = mubar
      input = mu
    []
    [Mbarmubar]
      type = ParsedCompute
      buffer = Mbarmubar
      enable_jit = true
      expression = 'Mbar*mubar' # (-M k^2) * μbar
      inputs = 'Mbar mubar'
    []
    [cbar]
      type = ForwardFFT
      buffer = cbar
      input = c
    []
    [rhsbar]
      type = ParsedCompute
      buffer = rhsbar
      enable_jit = true
      # RHS in reciprocal space: cbar_t = Mbarmubar + kappabarbar * cbar
      expression = 'Mbarmubar + kappabarbar*cbar'
      inputs = 'Mbarmubar kappabarbar cbar'
    []
  []
[]

[TensorSolver]
  type = RungeKuttaChebyshev
  buffer = c
  reciprocal_buffer = cbar
  time_derivative_reciprocal = rhsbar
  substeps = 500
  stages = 10
  damping = 0.05
[]

[AuxVariables]
  [mu]
    family = MONOMIAL
    order = CONSTANT
  []
  [c]
  []
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
  [C]
    type = ElementIntegralVariablePostprocessor
    variable = c
    execute_on = 'TIMESTEP_END'
  []
  [time]
    type = TimePostprocessor
    execute_on = 'TIMESTEP_END'
  []
  [dt_pp]
    type = TimestepSize
    execute_on = 'TIMESTEP_END'
  []
[]

[Problem]
  type = TensorProblem
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
