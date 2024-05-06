start_x = 0

[Mesh]
  [slug]
    type = GeneratedMeshGenerator
    dim = 2
    nx = 20
    ny = 10
    xmin = ${fparse -1 - start_x}
    xmax = ${fparse -start_x}
    ymin = -0.15
    ymax = 0.15
  []
  [slug_block]
    type = SubdomainIDGenerator
    input = slug
    subdomain_id = 1
  []
  [slug_side]
    type = SideSetsAroundSubdomainGenerator
    input = slug_block
    normal = '1 0 0'
    new_boundary = 10
    block = 1
  []

  [anvil]
    type = GeneratedMeshGenerator
    dim = 2
    nx = 3
    ny = 10
    xmin = 0
    xmax = 3
    ymin = -5
    ymax = 5
  []
  [anvil_block]
    type = SubdomainIDGenerator
    input = anvil
    subdomain_id = 2
  []
  [anvil_side1]
    type = SideSetsAroundSubdomainGenerator
    input = anvil_block
    normal = '-1 0 0'
    new_boundary = 20
    block = 2
  []
  [anvil_side2]
    type = SideSetsAroundSubdomainGenerator
    input = anvil_side1
    normal = '1 0 0'
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

[Modules/TensorMechanics/DynamicMaster]
  [slug]
    strain = FINITE
    add_variables = true
    hht_alpha = 0.11
    newmark_beta = 0.25
    newmark_gamma = 0.5
    density = 7750
    block = 1
    generate_output = stress_xx
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
    penalty = ${units 1000 GPa -> Pa}
  []
[]

[Materials]
  [slug_stress]
    type = ComputeFiniteStrainElasticStress
    block = 1
  []
  [anvil_stress]
    type = ComputeLinearElasticStress
    block = 2
  []
  [elasticity_slug]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = ${units 70 GPa -> Pa}
    poissons_ratio = 0.28
    block = 1
  []
  [elasticity_anvil]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = ${units 215 GPa -> Pa}
    poissons_ratio = 0.28
    block = 2
  []
[]

[ICs]
  [vel_x]
    type = ConstantIC
    variable = vel_x
    value = 10
  []
[]

[BCs]
  [fix_anvil_x]
    type = DirichletBC
    variable = disp_x
    value = 0
    boundary = 30
  []
  [fix_anvil_y]
    type = DirichletBC
    variable = disp_y
    value = 0
    boundary = 30
  []
[]

[Postprocessors]
  [vel_x]
    type = ElementAverageValue
    variable = vel_x
    block = 1
  []
[]

[Executioner]
  type = Transient
  solve_type = NEWTON
  dt = 1e-6
  num_steps = 1000
[]

[Outputs]
  exodus = true
  csv = true
  print_linear_residuals = false
  interval = 4
[]
