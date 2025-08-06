w=6

[Domain]
  dim = 2
  nx = 40
  ny = 40
  xmax = ${fparse w*pi*2}
  ymax = ${fparse w*pi*2/sin(pi/3)}
  mesh_mode = DOMAIN
[]

[AuxVariables]
  [phi]
  []
[]

[Outputs]
  exodus = false
[]

[TensorBuffers]
  [psi]
    map_to_aux_variable = phi
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
[]

crystal = '-(sin(sin(a)*y/2+cos(a)*x/2)^2 + sin(sin(a+1/3*pi)*y/2+cos(a+1/3*pi)*x/2)^2 + sin(sin(a-1/3*pi)*y/2+cos(a-1/3*pi)*x/2)^2 - 1.5)*0.25'
[Functions]
  [grain1]
    type = ParsedFunction
    expression = 'a := 0; ${crystal}'
  []
  [grain2]
    type = ParsedFunction
    expression = 'a := 0.95; ${crystal}'
  []
  [domain]
    type = ParsedFunction
    expression = 'r := (x-${w}*pi)^2+(y-${w}*pi)^2; if(r<(${w}*2/3*pi)^2, grain2, grain1)'
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
[]

[TensorSolver]
  type = SecantSolver
  buffer = psi
  substeps = 1
  absolute_tolerance = 1e-06
  relative_tolerance = 1e-05
  reciprocal_buffer = psibar
  linear_reciprocal = linear
  nonlinear_reciprocal = psi3bar
  adaptive_damping = true
  adaptive_damping_cutback_factor = 0.9
  adaptive_damping_growth_factor= 1.3
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 10
  [TimeStepper]
    type = TensorSolveIterationAdaptiveDT
    dt = 1
    max_iterations = 400
    min_iterations = 100
    growth_factor = 1.4
    cutback_factor = 0.9
  []
  dtmax = 500
[]

[TensorOutputs]
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'psi'
    enable_hdf5 = true
    # Do not transpose output to avoid regolding the test. In practice the default
    # of transpose = true should always be used
    transpose = false
  []
[]

# for cut in 0.5 0.6 0.7 0.8 0.9; do for grow in 1.1 1.2 1.3 1.4 1.5; do echo $cut $grow $(../../../swift-opt -i rotating_grain_secant.i TensorSolver/verbose=true TensorSolver/adaptive_damping=true TensorSolver/adaptive_damping_cutback_factor=$cut TensorSolver/adaptive_damping_growth_factor=$grow |tail -n1 |cut -d' ' -f1) >> log; done; done
