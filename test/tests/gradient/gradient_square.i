[Domain]
  dim = 3
  nx = 40
  ny = 40
  nz = 40
  xmax = ${fparse pi*2}
  ymax = ${fparse pi*4}
  zmax = ${fparse pi*6}
  mesh_mode = DUMMY
  device_names = cpu
[]

[TensorBuffers]
  [s]
  []
  [grad_sq]
  []
  [c2]
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
      expression = 'sin(x)+sin(y)+sin(z)'
    []
    [cos2]
      type = ParsedCompute
      buffer = c2
      extra_symbols = true
      expression = 'cos(x)^2+cos(y)^2+cos(z)^2'
    []
    [grad_sq]
      type = FFTGradientSquare
      buffer = grad_sq
      input = s
    []
    [diff]
      type = ParsedCompute
      buffer = diff
      inputs = 'grad_sq c2'
      expression = 'abs(grad_sq - c2)'
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
