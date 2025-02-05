#
# Contact in RZ does not seem to work reliably, since our anvil is just a coordinate plane.
# Let's try using a dirichlet BC that is only activated if the node penetrates and snaps
# it back to the coordinate plane!
# THIS DOES NOT WORK
#

[Mesh]
  coord_type = RZ

  [slug]
    type = GeneratedMeshGenerator
    dim = 2
    nx = 10
    ny = 80
    # nx = 20
    # ny = 10
    xmin = 0
    xmax = ${units 0.25 in -> m}
    ymin = ${units -3 in -> m}
    ymax = 0
  []
  [slug_block]
    type = SubdomainIDGenerator
    input = slug
    subdomain_id = 1
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
    # type = TotalLagrangianStressDivergence
    variable = disp_x
    component = 0
    block = 1
  []
  [sdy]
    type = TotalLagrangianStressDivergenceAxisymmetricCylindrical
    # type = TotalLagrangianStressDivergence
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

[Materials]
  [slug_strain]
    type = ComputeLagrangianStrainAxisymmetricCylindrical
    # type = ComputeLagrangianStrain
    block = 1
  []
  [slug_stress]
    type = ComputeLagrangianLinearElasticStress
    block = 1
  []
  [elasticity_slug]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = ${units 70 GPa -> Pa}
    poissons_ratio = 0.28
    block = 1
  []
  [slug_density]
    type = Density
    density = ${units 2700 kg/m^3}
  []
[]

[ICs]
  [vel_y]
    type = ConstantIC
    variable = vel_y
    value = ${units 100 m/s}
    block = 1
  []
[]

[BCs]
  [anvil]
    type = CoordinatePlaneNodalContactBC
    variable = disp_y
    boundary = top
    obstacle = POSITIVE
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
  [gap_y]
    type = SideAverageValue
    variable = disp_y
    boundary = top
  []
[]

[Executioner]
  type = Transient
  solve_type = NEWTON
  dt = ${units 1e-7 s}
  num_steps = 2000
[]

[Outputs]
  exodus = true
  csv = true
  print_linear_residuals = false
  perf_graph = true
  interval = 10
[]
