[Domain]
  dim = 2
  nx = 100
  ny = 100
  xmax = 100
  ymax = 100
  device_names = 'cpu'
  mesh_mode = MANUAL
[]

[Mesh]
  type = LatticeBoltzmannMesh
  dim = 2
  nx = 100
  ny = 100
  xmax = 100
  ymax = 100
[]

[Stencil]
  [d2q9]
    type = LBMD2Q9
  []
[]

[TensorBuffers]
  [rho]
    map_to_aux_variable = density
  []
  [u]
    vector_size = 2
    map_to_aux_variable = velocity
  []
  [u_magnitude]
    map_to_aux_variable = speed
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
    [equilibrium_2]
      type = LBMEquilibrium
      buffer = f_post_collision
      rho = rho
      velocity = u
    []
    [equilibrium_3]
      type = LBMEquilibrium
      buffer = feq
      rho = rho
      velocity = u
    []
  []
  [Solve]
    [Collision]
      type = LBMRegularizedMRTCollision
      buffer = f_post_collision
      f = f
      feq = feq
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
      # body_force = 0.000001
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
  # Any boundary that is not specified will be periodic
  [Boundary]
    [left]
      type = LBMFixedVelocityBC2D
      buffer = f
      f = f
      velocity = 0.1
      boundary = left
    []
    [right]
      type = LBMFixedVelocityBC2D
      buffer = f
      f = f
      velocity = 0.09999
      boundary = right
    []
    [front]
      type = LBMBounceBack
      buffer = f
      f_old = f_post_collision
      boundary = front
    []
    [back]
      type = LBMBounceBack
      buffer = f
      f_old = f_post_collision
      boundary = back
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

[AuxVariables]
  [density]
    family = MONOMIAL
    order = CONSTANT
  []
  [speed]
    family = MONOMIAL
    order = CONSTANT
  []
  [velocity]
    family = MONOMIAL
    order = CONSTANT
    components = 2
  []
[]

[AuxKernels]
  [density]
    type = ProjectTensorAux
    buffer = rho
    variable = density
    execute_on = timestep_end
  []
  [speed]
    type = ProjectTensorAux
    buffer = u_magnitude
    variable = speed
    execute_on = timestep_end
  []
  [velocity]
    type = ProjectVectorTensorAux
    buffer = u
    variable = velocity
    execute_on = timestep_end
  []
[]

[Problem]
  type = LatticeBoltzmannProblem
  print_debug_output = true
  substeps = 50
[]

[Executioner]
  type = Transient
  num_steps = 100
[]

[Outputs]
  exodus = true
[]
