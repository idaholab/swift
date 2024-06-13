[Mesh]
  type = FFTMesh
  dim = 2
  nx = 50
  ny = 50
[]

[FFTBuffers]
  [eta]
  []
  [f]
  []
[]

[Problem]
  type = FFTProblem
[]

[Executioner]
  type = Transient
  num_steps = 4
[]
