#
# Simple Cahn-Hilliard solve on a 2D grid. We create a matching (conforming)
# MOOSE mesh (with one element per FFT grid cell) and project the solution onto
# the MOOSE mesh to utilize the exodus output object.
#

[Domain]
  dim = 2
  nx = 20
  ny = 20
  xmax = 3
  ymax = 3
  mesh_mode = DOMAIN
  device_names = cpu
[]


[TensorBuffers]
  [c]
  []
  [cbar]
  []
  [mu]
  []
  [mubar]
  []
  [Mbarmubar]
  []
  # constant tensors
  [Mbar]
  []
  [kappabarbar]
  []
[]

[TensorComputes]
  [Initialize]
    [c]
      # Random initial condition around a concentration of 1/2
      type = RandomTensor
      buffer = c
      min = 0.44
      max = 0.56
      seed = 0
    []
    [mu_init]
      type = ConstantTensor
      buffer = mu
    []

    # precompute fixed factors for the solve
    [Mbar]
      type = ReciprocalLaplacianFactor
      factor = 0.2 # Mobility
      buffer = Mbar
    []
    [kappabarbar]
      type = ReciprocalLaplacianSquareFactor
      factor = -0.001 # kappa
      buffer = kappabarbar
    []
  []

  [Solve]
    [mu]
      type = ParsedCompute
      buffer = mu
      enable_jit = true
      expression = '0.1*c^2*(c-1)^2'
      derivatives = c
      inputs = c
    []
    [mubar]
      type = ForwardFFT
      buffer = mubar
      input = mu
    []
    [Mbarmubar]
      type = ParsedCompute
      buffer = Mbarmubar
      enable_jit = true
      expression = 'Mbar*mubar'
      inputs = 'Mbar mubar'
    []
    [cbar]
      type = ForwardFFT
      buffer = cbar
      input = c
    []

    # root compute
    [cahn_hilliard]
      type = ComputeGroup
      computes = 'mu mubar Mbarmubar cbar'
    []
  []
[]

[TensorSolver]
  type = SemiImplicitSolver
  root_compute = cahn_hilliard
  buffer = c
  reciprocal_buffer = cbar
  linear_reciprocal = kappabarbar
  nonlinear_reciprocal = Mbarmubar
  substeps = 10
[]

[AuxVariables]
  [mu]
    # the mu tensor  is projected onto this elemental variable
    family = MONOMIAL
    order = CONSTANT
  []
  [c]
    # the c tensor is projected onto this nodal variable
  []
[]

[AuxKernels]
  active = ''
  [c]
    type = ProjectTensorAux
    buffer = c
    variable = c
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [mu]
    type = ProjectTensorAux
    buffer = mu
    variable = mu
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[Postprocessors]
  [min_c]
    type = SemiImplicitCriticalTimeStep
    buffer = kappabarbar
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [delta_int_c]
    type = TensorIntegralChangePostprocessor
    buffer = c
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 10
  dt = 1e-3
[]

[TensorOutputs]
  active = ''
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'c mu'
    output_mode = 'Node Cell'
    enable_hdf5 = true
  []
[]

[Outputs]
  exodus = true
  csv = true
[]
