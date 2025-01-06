[Domain]
  dim = 2
  nx = 50
  ny = 50
  device_names = 'cpu'

  mesh_mode = MANUAL
[]

[Mesh]
  type = LatticeBoltzmannMesh
  dim = 2
  nx = 50
  ny = 50
[]

[Stencil]
  [d2q9]
    type = LBMD2Q9
  []
[]

[TensorBuffers]
  [f]
  []
[]

[Problem]
  type = LatticeBoltzmannProblem
  print_debug_output = true
  spectral_solve_substeps = 1
[]

[Executioner]
  type = Transient
  num_steps = 1
[]

