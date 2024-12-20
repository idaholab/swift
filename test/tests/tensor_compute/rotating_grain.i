[Domain]
  dim = 2
  nx = 400
  ny = 400
  xmax = ${fparse pi*2*30}
  ymax = ${fparse pi*2*30}

  device_names = 'cuda'

  mesh_mode = DOMAIN
[]

[TensorBuffers]
  [psi]
    map_to_aux_variable = psi
  []
  [psibar]
  []
  [psi3]
  []
  [psi3bar]
  []
  # constant tensors
  [linear]
  []

  # output
  [filter]
    map_to_aux_variable = filter
  []
  [filterbar]
  []
[]

[AuxVariables]
  [psi]
  []
  [filter]
  []
[]

[Functions]
  [grain1]
    type = ParsedFunction
    # expression = 'r := (x-30*pi)^2+(y-30*pi)^2; if(r<15^2, (sin((x+y)/1.41)*cos((x-y)/1.41))^2, (sin(x)*cos(y))^2)'
    expression = 'a := 0; -sin(sin(a)*y+cos(a)*x)^2*sin(sin(a+1/3*pi)*y+cos(a+1/3*pi)*x)^2*sin(sin(a-1/3*pi)*y+cos(a-1/3*pi)*x)^2'
  []
  [grain2]
    type = ParsedFunction
    # expression = 'r := (x-30*pi)^2+(y-30*pi)^2; if(r<15^2, (sin((x+y)/1.41)*cos((x-y)/1.41))^2, (sin(x)*cos(y))^2)'
    expression = 'a := 0.95; -sin(sin(a)*y+cos(a)*x)^2*sin(sin(a+1/3*pi)*y+cos(a+1/3*pi)*x)^2*sin(sin(a-1/3*pi)*y+cos(a-1/3*pi)*x)^2'
  []
  [domain]
    type = ParsedFunction
    expression = 'r := (x-30*pi)^2+(y-30*pi)^2; if(r<(20*pi)^2, grain2, grain1)'
    symbol_names = 'grain1 grain2'
    symbol_values = 'grain1 grain2'
  []
[]

[TensorComputes]
  [Initialize]
    [psi]
      type = MooseFunctionTensor
      buffer = psi
      function = domain
    []
    [linear]
      type = SwiftHohenbergLinear
      buffer = linear
      alpha = 1
      r = 0.025
    []
  []

  [Solve]
    [psi3]
      type = ParsedCompute
      buffer = psi3
      enable_jit = true
      expression = "0.20*psi^2-psi^3"
      inputs = psi
    []
    [psibar]
      type = ForwardFFT
      buffer = psibar
      input = psi
    []
    [psi3bar]
      type = ForwardFFT
      buffer = psi3bar
      input = psi3
    []
  []

  [Postprocess]
    [low_pass]
      type = ParsedCompute
      buffer = filterbar
      extra_symbols = true
      expression = 'psibar * exp(-k2*10)'
      inputs = psibar
    []
    [filter]
      type = InverseFFT
      buffer = filter
      input = filterbar
    []
  []
[]

[TensorSolver]
  type = SemiImplicitSolver
  buffer = psi
  reciprocal_buffer = psibar
  linear_reciprocal = linear
  nonlinear_reciprocal = psi3bar
[]

[Problem]
  type = TensorProblem
  spectral_solve_substeps = 110
[]

[Executioner]
  type = Transient
  num_steps = 120
  [TimeStepper]
    type = IterationAdaptiveDT
    growth_factor = 1.2
    dt = 1.7
  []
  dtmax = 500
[]

[Postprocessors]
  [min_psi]
    type = TensorExtremeValuePostprocessor
    buffer = psi
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_psi]
    type = TensorExtremeValuePostprocessor
    buffer = psi
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
  [Psi]
    type = TensorIntegralPostprocessor
    buffer = psi
  []
[]

# [TensorOutputs]
#   [xdmf]
#     type = XDMFTensorOutput
#     buffer = 'psi'
#     output_mode = 'Node'
#     enable_hdf5 = true
#   []
# []

[Outputs]
  exodus = true
  perf_graph = true
  execute_on = 'TIMESTEP_END'
[]
