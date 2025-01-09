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
[]

[TensorComputes]
  [Initialize]
    [rho]
      type = LBMConstantTensor
      buffer = rho
      value = 1.0
    []
    [u]
      type = LBMConstantTensor
      buffer = u
      value = 0.001
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

