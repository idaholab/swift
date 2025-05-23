[Domain]
  dim = 2
  nx = 32
  ny = 32
  xmax = 32
  ymax = 32
  device_names = 'cpu'
  mesh_mode = MANUAL
[]

[Mesh]
  type = LatticeBoltzmannMesh
  dim = 2
  nx = 32
  ny = 32
  xmax = 32
  ymax = 32
  load_mesh_from_vtk=true
  mesh_file="examples/lbm/data/porous_media.vts"
[]

[Stencil]
  [d2q9]
    type = LBMD2Q9
  []
[]

[TensorBuffers]
  [rho]
    type=LBMTensorBuffer
  []
  [u]
    type=LBMTensorBuffer
    dimension=2
  []
  [u_magnitude]
    type = LBMTensorBuffer
  []
  [f]
    type=LBMTensorBuffer
    dimension=9
  []
  [f_post_collision]
    type=LBMTensorBuffer
    dimension=9
  []
  [feq]
    type=LBMTensorBuffer
    dimension=9
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
    [front]
    type = LBMFixedPressureBC2D
      buffer = f
      f = f
      density = 1.0
      boundary = top
    []
    [back]
      type = LBMFixedPressureBC2D
      buffer = f
      f = f
      density = 0.9999
      boundary = bottom
    []
    [wall]
      type = LBMBounceBack
      buffer = f
      f_old = f_post_collision
      boundary = wall
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

[Problem]
  type = LatticeBoltzmannProblem
  print_debug_output = true
  substeps = 1
  tolerance = 1.0e-10
[]

[Executioner]
  type = Transient
  num_steps = 2
[]

[TensorOutputs]
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'rho'
    output_mode = 'Cell'
    enable_hdf5 = true
  []
 
  [xdmf2]
    # second output to trigger the hdf5 thread safety error
    type = XDMFTensorOutput
    buffer = 'u_magnitude'
    output_mode = 'Cell'
    enable_hdf5 = true
  []
[]
