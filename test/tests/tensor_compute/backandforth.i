[Domain]
  xmax = ${fparse pi*4}
  ymax = ${fparse pi*4}
  device_names = 'cpu'
  mesh_mode = DUMMY
[]

[TensorBuffers]
  [eta_gold]
  []
  [eta]
  []
  [eta_bar]
  []
  [eta2]
  []
  [zero]
  []
  [diff]
  []
[]

[TensorComputes]
  [Initialize]
    [eta_gold]
      type = ParsedCompute
      buffer = eta_gold
      expression = 'sin(x)+sin(y)+sin(z)'
      extra_symbols = true
    []
    [eta]
      type = ParsedCompute
      buffer = eta
      expression = eta_gold
      inputs = eta_gold
    []
    [eta2]
      type = ConstantTensor
      buffer = eta2
      real = 1
    []
    [zero]
      type = ConstantReciprocalTensor
      buffer = zero
      real = 0
      imaginary = 0
    []
  []

  [Solve]
    [eta_bar]
      type = ForwardFFT
      buffer = eta_bar
      input = eta
    []
    [eta_2]
      type = InverseFFT
      buffer = eta2
      input = eta_bar
    []
  []

  [Postprocess]
    [diff]
      type = ParsedCompute
      buffer = diff
      expression = 'abs(eta - eta2) + abs(eta - eta_gold)'
      inputs = 'eta eta2 eta_gold'
    []
  []
[]

[Postprocessors]
  [norm]
    type = TensorIntegralPostprocessor
    buffer = diff
  []
[]

[TensorSolver]
  type = SemiImplicitSolver
  buffer = eta
  reciprocal_buffer = eta_bar
  linear_reciprocal = zero
  nonlinear_reciprocal = zero
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 4
[]

[Outputs]
  csv = true
[]
