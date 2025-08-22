[Domain]
  dim = 3
  nx = 10
  ny = 10
  nz = 10
  mesh_mode = DUMMY
[]

[Stencil]
  [d3q27]
    type = LBMD3Q27
  []
[]

[TensorBuffers]
  [f]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [f_bounce_back]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [velocity]
    type=LBMTensorBuffer
    buffer_type = mv
  []
  [density]
    type=LBMTensorBuffer
    buffer_type = ms
  []
[]

[TensorComputes]
  [Initialize]
    [initial_density]
      type = LBMConstantTensor
      buffer = density
      constants = 1.0
    []
    [initial_velocity]
      type = LBMConstantTensor
      buffer = velocity
      constants = '0.0001 0.0005 0.0'
    []
    [initial_f]
      type = LBMEquilibrium
      buffer = f
      bulk = density
      velocity = velocity
    []
  []
  [Solve]
    [density]
      type = LBMComputeDensity
      buffer = density
      f = f
    []
    [velocity]
      type = LBMComputeVelocity
      buffer = velocity
      f = f
      rho = density
      add_body_force = true
      body_force_x = 0.0001
    []
  []
  [Boundary]
    [left]
      type = LBMFixedZerothOrderBC27Q
      buffer = f
      f = f
      value = 1.1
      boundary = left
    []
    [right]
      type = LBMFixedFirstOrderBC27Q
      buffer = f
      f = f
      value = 0.0001
      boundary = right
    []
  []
[]

[TensorSolver]
  type = LBMStream
  buffer = f
  f_old = f
[]

[Problem]
  type = LatticeBoltzmannProblem
  substeps = 2
[]

[Executioner]
  type = Transient
  num_steps = 2
[]

[TensorOutputs]
  [xdmf2]
    type = XDMFTensorOutput
    buffer = 'velocity density'
    output_mode = 'Cell Cell'
    enable_hdf5 = true
  []
[]
