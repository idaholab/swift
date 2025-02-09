[Mesh]
  [gen]
    type = GeneratedMeshGenerator
    dim = 3
    nx = 5
    ny = 5
    nz = 5
  []
[]

[Functions]
  [a]
    type = ParsedFunction
    expression = 'x+y^2+sqrt(z)+cos(3*t)'
  []

  dx=0.1
  dy=0.2
  dz=0.3
  dt=0.4

  [b]
    type = ShiftedFunction
    function = a
    shift = '${dx} ${dy} ${dz}'
    delta_t = ${dt}
  []

  [c]
    type = ParsedFunction
    expression = 'abs((x+${dx})+(y+${dy})^2+sqrt(z+${dz})+cos(3*(t+${dt}))-b)'
    symbol_names = b
    symbol_values = b
  []
[]

[Postprocessors]
  [C]
    type = FunctionElementIntegral
    function = c
  []
[]

[Problem]
  solve = false
[]

[Executioner]
  type = Transient
  dt = 0.15
  num_steps = 10
[]

[Outputs]
  csv = true
[]
