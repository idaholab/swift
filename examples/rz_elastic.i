start_y = 0

[Mesh]
  coord_type = RZ

  [slug]
    type = GeneratedMeshGenerator
    dim = 2
    nx = 10
    ny = 40
    # nx = 20
    # ny = 10
    xmin = 0
    xmax = ${units 0.25 in -> cm}
    ymin = ${fparse -1 - start_y}
    ymax = ${fparse -start_y}
  []
  [slug_block]
    type = SubdomainIDGenerator
    input = slug
    subdomain_id = 1
  []
  [slug_side]
    type = SideSetsAroundSubdomainGenerator
    input = slug_block
    normal = '0 1 0'
    new_boundary = 10
    block = 1
  []

  [anvil]
    type = GeneratedMeshGenerator
    dim = 2
    nx = 5
    ny = 3
    xmin = 0
    xmax = ${units 2 in -> cm}
    ymin = 0
    ymax = 3
  []
  [anvil_block]
    type = SubdomainIDGenerator
    input = anvil
    subdomain_id = 2
  []
  [anvil_side1]
    type = SideSetsAroundSubdomainGenerator
    input = anvil_block
    normal = '0 -1 0'
    new_boundary = 20
    block = 2
  []
  [anvil_side2]
    type = SideSetsAroundSubdomainGenerator
    input = anvil_side1
    normal = '0 1 0'
    new_boundary = 30
    block = 2
  []

  [combine]
    type = MeshCollectionGenerator
    inputs = 'anvil_side2 slug_side'
  []
[]

[GlobalParams]
  displacements = 'disp_x disp_y'
  large_kinematics = true
[]

[Problem]
  # type = AugmentedLagrangianContactFEProblem
  kernel_coverage_check = false
[]

[Variables]
  [disp_x]
  []
  [disp_y]
  []
[]

[AuxVariables]
  [vel_x]
    block = 1
  []
  [vel_y]
    block = 1
  []
  [accel_x]
    block = 1
  []
  [accel_y]
    block = 1
  []
  [T]
    initial_condition = ${units 25 degC}
  []
[]

[Kernels]
  [sdx]
    type = TotalLagrangianStressDivergenceAxisymmetricCylindrical
    variable = disp_x
    component = 0
    block = 1
  []
  [sdy]
    type = TotalLagrangianStressDivergenceAxisymmetricCylindrical
    variable = disp_y
    component = 1
    block = 1
  []

  [ifx]
    type = InertialForce
    variable = disp_x
    velocity = vel_x
    acceleration = accel_x
    beta = 0.25
    gamma = 0.5
    alpha = 0 # not implemented in TotalLagrangianStressDivergence
    block = 1
  []
  [ify]
    type = InertialForce
    variable = disp_y
    velocity = vel_y
    acceleration = accel_y
    beta = 0.25
    gamma = 0.5
    alpha = 0 # not implemented in TotalLagrangianStressDivergence
    block = 1
  []
[]

[AuxKernels]
  [vel_x]
    type = NewmarkVelAux
    variable = vel_x
    acceleration = accel_x
    gamma = 0.5
    execute_on = 'TIMESTEP_END'
  []
  [vel_y]
    type = NewmarkVelAux
    variable = vel_y
    acceleration = accel_y
    gamma = 0.5
    execute_on = 'TIMESTEP_END'
  []

  [accel_x]
    type = NewmarkAccelAux
    variable = accel_x
    displacement = disp_x
    velocity = vel_x
    beta = 0.25
    execute_on = 'TIMESTEP_END'
  []
  [accel_y]
    type = NewmarkAccelAux
    variable = accel_y
    displacement = disp_y
    velocity = vel_y
    beta = 0.25
    execute_on = 'TIMESTEP_END'
  []
[]

[Modules/TensorMechanics/Master]
  [anvil]
    strain = SMALL
    block = 2
    generate_output = stress_xx
  []
[]

[Contact]
  [impact]
    secondary = 10
    primary = 20
    model = frictionless
    # correct_edge_dropping = true
    formulation = mortar_penalty
    # al_penetration_tolerance = 1e-6
    # c_normal = 1e+2
    penalty = ${units 1e+12 Pa -> N/cm^2}
  []
[]

[Materials]
  [slug_strain]
    type = ComputeLagrangianStrainAxisymmetricCylindrical
    block = 1
  []
  [slug_stress]
    type = ComputeLagrangianLinearElasticStress
    block = 1
  []
  [elasticity_slug]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = ${units 70 GPa -> N/cm^2}
    poissons_ratio = 0.28
    block = 1
  []
  [slug_density]
    type = Density
    density = ${units 2700 kg/m^3 -> g/cm^3}
  []

  [anvil_stress]
    type = ComputeLinearElasticStress
    block = 2
  []
  [elasticity_anvil]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = ${units 215 GPa -> N/cm^2}
    poissons_ratio = 0.28
    block = 2
  []
[]

[ICs]
  [vel_y]
    type = ConstantIC
    variable = vel_y
    value = ${units 100 m/s -> cm/ms}
    block = 1
  []
[]

[BCs]
  [fix_anvil_x]
    type = DirichletBC
    variable = disp_x
    value = 0
    boundary = 'left 30'
  []
  [fix_anvil_y]
    type = DirichletBC
    variable = disp_y
    value = 0
    boundary = 30
  []
  [symmetry]
    type = DirichletBC
    variable = disp_x
    value = 0
    boundary = left
  []
[]

[Postprocessors]
  [vel_y]
    type = ElementAverageValue
    variable = vel_y
    block = 1
  []
[]

[Executioner]
  type = Transient
  solve_type = NEWTON
  dt = ${units 1e-6 s -> ms}
  num_steps = 2000
[]

[Outputs]
  exodus = true
  csv = true
  print_linear_residuals = false
  perf_graph = true
  interval = 10
[]
