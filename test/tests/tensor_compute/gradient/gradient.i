[Domain]
  dim = 2
  nx = 40
  ny = 40
  xmax = ${fparse pi*2}
  ymax = ${fparse pi*20}
  mesh_mode = DUMMY
  device_names = cpu
[]

[TensorBuffers]
  [s]
  []
  [grad_s]
  []
  [c]
  []
  [diff]
  []
[]

[TensorComputes]
  [Initialize]
    [sin]
      type = ParsedCompute
      buffer = s
      extra_symbols = true
      expression = 'sin(x)'
    []
    [cos]
      type = ParsedCompute
      buffer = c
      extra_symbols = true
      expression = 'cos(x)'
    []
    [grad_sin]
      type = FFTGradient
      buffer = grad_s
      input = s
      direction = x
    []
    [diff]
      type = ParsedCompute
      buffer = diff
      inputs = 'grad_s c'
      expression = 'grad_s - c'
    []
  []
[]

[Postprocessors]
  [diff]
    type = TensorIntegralPostprocessor
    buffer = diff
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 1
[]

[Outputs]
  csv = true
[]
