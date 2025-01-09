[Domain]
  dim = 2
  nx = 50
  ny = 50
  xmax = 50
  ymax = 50
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
    vector_size = 9
  []
  [f_post_collision]
    vector_size = 9
  []
  [feq]
    vector_size = 9
  []
  [rho]
  []
  [u]
    vector_size = 2
  []
[]

[TensorComputes]
  [Initialize]
    [rho]
      type = ConstantTensor
      buffer =rho
      real = 1.0
      full = true
    []
    [u]
      type = ConstantTensor
      buffer = u
      real = 0.001
      full = true
    []
    [feq]
      type = LBMEquilibrium
      buffer = feq
      rho = rho
      velocty = u
    []
    [f]
      type = LBMEquilibrium
      buffer = f
      rho = rho
      velocty = u
    []
    [f_post_collision]
      type = LBMEquilibrium
      buffer = f_post_collision
      rho = rho
      velocty = u
    []
  []
  [Solve]
    [Collision]
      type = LBMBGKCollision
      buffer = f_post_collision
      f = f
      feq = feq
      tau = 0.8
    []
  []
[]

[Postprocessors]
  [rho_avg]
    type = TensorAveragePostprocessor
    buffer = rho
    execute_on = 'TIMESTEP_BEGIN'
  []
[]

[Problem]
  type = LatticeBoltzmannProblem
  print_debug_output = true
  spectral_solve_substeps = 2
[]

[Executioner]
  type = Transient
  num_steps = 1
[]

