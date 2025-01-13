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
  [u_magnitude]
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
    [velocity]
      type = LBMConstantTensor
      buffer = u
      value = 0.000
    []
    [speed]
      type = LBMConstantTensor
      buffer = u_magnitude
      value = 0
    []
    [equilibrium]
      type = LBMEquilibrium
      buffer = f
      rho = rho
      velocity = u
    []
  []
  [Solve]
    [Collision]
      type = LBMMRTCollision
      buffer = f_post_collision
      f = f
      feq = feq
      # tau = 0.8
    []
    [Equilibrium]
      type = LBMEquilibrium
      buffer = feq
      rho = rho
      velocity = u
    []
    [Density]
      type = LBMComputeDensity
      buffer = rho
      f = f
    []
    [Velocity]
      type = LBMComputeVelocity
      buffer = u
      f = f
      rho = rho 
      body_force = 0.0001
    []
    [Spped]
      type = LBMComputeVelocityMagnitude
      buffer = u_magnitude
      velocity = u
    []
    [Residual]
      type = LBMComputeResidual
      speed = u_magnitude
      # TODO this buffer is redundant, but avoids 'missing parameter' error
      buffer = u_magnitude
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
  [speed_avg]
    type = TensorAveragePostprocessor
    buffer = u_magnitude
    execute_on = 'TIMESTEP_BEGIN'
  []
[]

[Problem]
  type = LatticeBoltzmannProblem
  print_debug_output = true
  spectral_solve_substeps = 1
[]

[Executioner]
  type = Transient
  num_steps = 10
[]
