[Domain]
  dim = 1
  nx = 1000
  xmax = ${fparse pi*2}
  mesh_mode = DUMMY
[]

# viscosity coefficient
nu=1e-4

[TensorComputes]
  [Initialize]
    [x0]
      type = ParsedCompute
      buffer = u
      extra_symbols = true
      expression = sin(x)^3
    []
    [nuk2]
      type = ReciprocalLaplacianSquareFactor
      buffer = nuk2
      factor = -${nu}
    []

    [mms_solution0]
      type = ParsedCompute
      buffer = u_solution
      extra_symbols = true
      expression = sin(x)^3*exp(-t)
    []

    [stabilization]
      type = ParsedCompute
      buffer = S
      extra_symbols = true
      expression = -0.0008*k2
    []
  []

  [Solve]
    [mms_solution]
      type = ParsedCompute
      buffer = u_solution
      extra_symbols = true
      expression = sin(x)^3*exp(-t)
    []
    [mms_source_term]
      type = ParsedCompute
      buffer = u_forcing
      extra_symbols = true
      expression = '((9*${nu}*sin(x)^2 - 6*${nu} - sin(x)^2)*exp(t) + 3*sin(x)^4*cos(x)) * exp(-2*t) * sin(x)'
    []

    [grad_u]
      type = FFTGradient
      direction = X
      input = u
      buffer = grad_u
    []
    [F_u]
      type = ForwardFFT
      buffer = F_u
      input = u
    []
    [u_grad_u]
      type = ParsedCompute
      expression = '-u * grad_u + u_forcing'
      inputs = 'u grad_u u_forcing'
      buffer = u_grad_u
    []
    [F_u_grad_u]
      type = ForwardFFT
      buffer = F_u_grad_u
      input = u_grad_u
    []

    [stabilize]
      type = ApplyStabilization
      linear = nuk2
      nonlinear = F_u_grad_u
      reciprocal = F_u
      stabilization = S
      stabilized_linear = L
      stabilized_nonlinear = N
    []
  []

  [Postprocess]
    [diff]
      type = ParsedCompute
      buffer = diff
      expression = 'u - u_solution'
      inputs = 'u u_solution'
    []
    [l2]
      type = ParsedCompute
      buffer = l2
      expression = 'diff^2'
      inputs = diff
    []
  []
[]

# [TensorSolver]
#   type = AdamsBashforthMoulton
#   buffer = u
#   substeps = 25
#   reciprocal_buffer = F_u
#   linear_reciprocal = nuk2
#   nonlinear_reciprocal = F_u_grad_u
#   corrector_steps = 0
# []

# [TensorSolver]
#   type = AdamsBashforthMoulton
#   buffer = u
#   substeps = 8
#   reciprocal_buffer = F_u
#   linear_reciprocal = nuk2
#   nonlinear_reciprocal = F_u_grad_u
#   corrector_steps = 1
# []

[TensorSolver]
  type = AdamsBashforthMoulton
  buffer = u
  substeps = 5
  reciprocal_buffer = F_u
  linear_reciprocal = L
  nonlinear_reciprocal = N
  corrector_order = 3
  predictor_order = 2
  corrector_steps = 3
  verbose = true
[]

# [TensorSolver]
#   type = AdamsBashforthMoulton
#   buffer = u
#   substeps = 50
#   reciprocal_buffer = F_u
#   linear_reciprocal = nuk2
#   nonlinear_reciprocal = F_u_grad_u
#   corrector_steps = 0
# []

# [TensorSolver]
#   type = SecantSolver
#   buffer = u
#   substeps = 1
#   absolute_tolerance = 1e-06
#   relative_tolerance = 1e-05
#   dt_epsilon = 1e-1
#   reciprocal_buffer = F_u
#   linear_reciprocal = nuk2
#   nonlinear_reciprocal = F_u_grad_u
#   adaptive_damping = true
#   damping = 1
#   trust_radius = 10.0
#   adaptive_damping_cutback_factor = 0.5
#   adaptive_damping_growth_factor= 1.2
#   verbose = true
# []

# [TensorSolver]
#   type = BroydenSolver
#   buffer = u
#   substeps = 1
#   absolute_tolerance = 1e-06
#   relative_tolerance = 1e-05
#   dt_epsilon = 1e-4
#   reciprocal_buffer = F_u
#   linear_reciprocal = nuk2
#   nonlinear_reciprocal = F_u_grad_u
#   initial_jacobian_guess = 1
#   # adaptive_damping = true
#   damping = 0.1
#   # trust_radius = 10.0
#   # adaptive_damping_cutback_factor = 0.5
#   # adaptive_damping_growth_factor= 1.2
#   verbose = true
#   max_iterations =20
# []

# [TensorSolver]
#   type = AndersonSolver
#   buffer = u
#   substeps = 1
#   absolute_tolerance = 1e-06
#   relative_tolerance = 1e-05
#   reciprocal_buffer = F_u
#   linear_reciprocal = nuk2
#   nonlinear_reciprocal = F_u_grad_u
#   damping = 0.1
#   verbose = true
#   max_iterations =20
# []

# [VectorPostprocessors]
#   [csv]
#     type = TensorVectorExtract
#   []
# []

[Postprocessors]
  [l2]
    type = TensorIntegralPostprocessor
    buffer = l2
    # execute_on = 'INITIAL TIMESTEP_END'
    execute_on = 'FINAL'
  []
[]

[Executioner]
  type = Transient
  num_steps = 10
  # [TimeStepper]
  #   type = TensorSolveIterationAdaptiveDT
  #   dt = 1
  #   max_iterations = 400
  #   min_iterations = 100
  #   growth_factor = 1.4
  #   cutback_factor = 0.9
  # []
  # dtmax = 500
  dt = 10
[]

[Outputs]
  [console]
    type = Console
    execute_postprocessors_on = 'FINAL'
  []
[]
