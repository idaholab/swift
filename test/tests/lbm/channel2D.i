[Domain]
  dim = 2
  nx = 5
  ny = 5
  xmax = 5
  ymax = 5
  device_names = 'cpu'

  mesh_mode = MANUAL
[]

[Mesh]
  type = LatticeBoltzmannMesh
  dim = 2
  nx = 5
  ny = 5
[]

[Stencil]
  [d2q9]
    type = LBMD2Q9
  []
[]

[TensorBuffers]
  [rho]
  []
  [u]
    vector_size = 2
  []
  [f]
    vector_size = 9
  []
  [f_post_collision]
    vector_size = 9
  []
  [feq]
    vector_size = 9
  []
[]

[TensorComputes]
  [Initialize]
    [density]
      type = LBMConstantTensor
      buffer = rho
      value = 1.0
    []
    [velocty]
      type = LBMConstantTensor
      buffer = u
      value = 0.001
    []
    [equilibrium]
      type = LBMEquilibrium
      buffer = feq
      rho = rho
      velocity = u
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
    [Equilibrium]
      type = LBMEquilibrium
      buffer = feq
      rho = rho
      velocity = u
    []
  []
[]

[TensorTimeIntegrators]
  [Stream]
    type = LBMStream
    buffer = f
    f_old = f_post_collision
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

