[Domain]
  dim = 2
  nx = 10
  ny = 2
  xmax = ${fparse pi*4}
  mesh_mode = DUMMY
  device_names = cpu
[]

[TensorBuffers]
  [c]
  []
[]

[TensorComputes]
  [Solve]
    [c]
      type = ParsedCompute
      buffer = c
      extra_symbols = true
      expression = sin(x+0.2*t)
    []
  []
[]

[Postprocessors]
  [v]
    type = TensorInterfaceVelocityPostprocessor
    buffer = c
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 10
  dt = 0.01
[]

[Outputs]
  csv = true
[]
