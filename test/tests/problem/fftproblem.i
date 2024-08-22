[Mesh]
  type = UniformTensorMesh
  dim = 2
  nx = 50
  ny = 50
[]

[TensorBuffers]
  [eta]
  []
  [f]
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 4
[]
