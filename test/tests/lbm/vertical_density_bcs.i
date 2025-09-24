[Domain]
  dim = 2
  nx = 10
  ny = 10
  mesh_mode = DUMMY
[]

[Stencil]
  [d2q9]
    type = LBMD2Q9
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
      constants = '0.0001 0.0005'
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
    []
  []
  [Boundary]
    [left]
      type = LBMBounceBack
      buffer = f
      f_old = f_bounce_back
      boundary = left
    []
    [right]
      type = LBMBounceBack
      buffer = f
      f_old = f_bounce_back
      boundary = right
    []
    [top]
      type = LBMFixedZerothOrderBC
      buffer = f
      f = f
      value = 1.1
      boundary = top
    []
    [bottom]
      type = LBMFixedZerothOrderBC
      buffer = f
      f = f
      value = 1.00000
      boundary = bottom
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
