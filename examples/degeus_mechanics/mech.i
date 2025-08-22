[Domain]
  dim = 3
  nx = 32
  ny = 32
  nz = 32
  xmax = ${fparse 2*pi}
  ymax = ${fparse 2*pi}
  zmax = ${fparse 2*pi}
  mesh_mode = DUMMY
  device_names = cpu
[]

[TensorComputes]
  [Initialize]
    [Finit]
      type = RankTwoDiagonalTensor
      buffer = F
      value = 1
    []
    [Stressinit]
      type = RankTwoDiagonalTensor
      buffer = stress
      value = 0
    []

    [phase]
      type = PhaseMechanicsTest
      buffer = phase
    []
    [K]
      type = ParsedCompute
      buffer = K
      expression = '(1-phase)*Ka + phase*Kb'
      inputs = phase
      constant_names = 'Ka Kb'
      constant_expressions = '0.833 8.33'
    []
    [mu]
      type = ParsedCompute
      buffer = mu
      expression = '(1-phase)*mua + phase*mub'
      inputs = phase
      constant_names = 'mua mub'
      constant_expressions = '0.386 3.86'
    []
  []

  [Solve]
    [hyper_elasticity]
      type = HyperElasticIsotropic
      buffer = stress
      F = Fnew
      K = K
      mu = mu
    []

    [root]
      [applied_strain]
        type = MacroscopicShearTensor
        buffer = applied_strain
      []
      [mech]
        type = FFTMechanics
        buffer = Fnew
        F = F
        K = K
        mu = mu
        l_tol = 1e-3
        l_max_its = 200
        nl_rel_tol = 2e-3
        nl_abs_tol = 2e-2
        constitutive_model = hyper_elasticity
        stress = stress
        applied_macroscopic_strain = applied_strain
        # hutchinson_steps = 64
        # jacobi_min_rel = 1e-2
        # jacobi_inv_cap = 1e4
        block_jacobi = true
        block_jacobi_damp=1e-1
        verbose = true
      []
    []
  []

  [Postprocess]
    [displacements]
      type = ComputeDisplacements
      buffer = disp
      F = F
    []
    [vonmises]
      type = ComputeVonMisesStress
      buffer = sV
    []
  []
[]

[TensorSolver]
  # no variables are integrated by this solver (FFTMechanics performs a steady state mechanics solve)
  type = ForwardEulerSolver
  root_compute = root
  # deformation tensor is just forwarded Fnew -> F
  forward_buffer = F
  forward_buffer_new = Fnew
  substeps = 1
[]

[TensorOutputs]
  [deformation_tensor]
    type = XDMFTensorOutput
    buffer = 'disp sV F'
    output_mode = 'OVERSIZED_NODAL CELL CELL'
    enable_hdf5 = true
  []
[]

[Executioner]
  type = Transient
  num_steps = 100
  dt = 0.01
[]

[Outputs]
  perf_graph = true
  console = true
[]
