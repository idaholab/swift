[Domain]
  dim = 3
  nx = 10
  ny = 10
  nz = 10
  mesh_mode = DUMMY
  device_names = cpu
[]

[TensorBuffers]
  [c]
  []
[]

[TensorComputes]
  [Initialize]
    [c]
      type = ParsedCompute
      buffer = c
      extra_symbols = true
      expression = '0.1*x^2+0.2*y^2+0.3*z^2'
    []
  []
[]

[VectorPostprocessors]
  [hist]
    type = TensorHistogram
    buffer = c
    bins = 20
    min = 0
    max = 1
    execute_on = 'TIMESTEP_END'
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
  execute_on = 'TIMESTEP_END'
[]
