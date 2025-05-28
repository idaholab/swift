[Domain]
  dim = 3
  nx = 40
  ny = 40
  nz = 40
  xmax = ${fparse pi*2}
  ymax = ${fparse pi*4}
  zmax = ${fparse pi*6}
  mesh_mode = DUMMY
[]

[TensorBuffers]
  [s]
  []
  [gradx_s]
  []
  [grady_s]
  []
  [gradz_s]
  []
  [cx]
  []
  [cy]
  []
  [cz]
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
    [cosx]
      type = ParsedCompute
      buffer = cx
      extra_symbols = true
      expression = 'cos(x)'
    []
    [cosy]
      type = ParsedCompute
      buffer = cy
      extra_symbols = true
      expression = 'cos(y)'
    []
    [cosz]
      type = ParsedCompute
      buffer = cz
      extra_symbols = true
      expression = 'cos(z)'
    []
    [gradx_sin]
      type = FFTGradient
      buffer = gradx_s
      input = s
      direction = x
    []
    [grady_sin]
      type = FFTGradient
      buffer = grady_s
      input = s
      direction = y
    []
    [gradz_sin]
      type = FFTGradient
      buffer = gradz_s
      input = s
      direction = z
    []
    [diff]
      type = ParsedCompute
      buffer = diff
      inputs = 'gradx_s grady_s gradz_s cx cy  cz'
      expression = 'abs(gradx_s - cx)+abs(grady_s - cy)+abs(gradz_s - cz)'
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
