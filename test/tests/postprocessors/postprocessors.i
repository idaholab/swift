[Domain]
  dim = 2
  nx = 40
  ny = 40
  xmax = 2
  ymax = 3
  mesh_mode = DUMMY
[]

[TensorBuffers]
  [c]
  []
  [c_bar]
  []
[]

[TensorComputes]
  [Initialize]
    [c]
      type = ParsedCompute
      buffer = c
      extra_symbols = true
      expression = -x+y+0.3
    []
    [c_bar]
      type = ForwardFFT
      buffer = c_bar
      input = c
    []
    [u]
      type = ConstantTensor
      buffer = u
      real = 0
    []
  []

  [Solve]
    [root]
      [test]
        type = ForwardFFT
        buffer = u_bar
        input = u
      []
    []
  []
[]

[TensorSolver]
  type = ForwardEulerSolver
  time_derivative_reciprocal = c_bar
  buffer = u
  reciprocal_buffer = u_bar
  substeps = 10
[]

[Postprocessors]
  [min_c]
    type = TensorExtremeValuePostprocessor
    buffer = c
    value_type = MIN
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [max_c]
    type = TensorExtremeValuePostprocessor
    buffer = c
    value_type = MAX
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [avg_c]
    type = TensorAveragePostprocessor
    buffer = c
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [int_c]
    type = TensorIntegralPostprocessor
    buffer = c
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [int_c_bar]
    type = ReciprocalIntegral
    buffer = c_bar
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [count]
    type = ComputeGroupExecutionCount
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 0
[]

[Outputs]
  csv = true
[]
