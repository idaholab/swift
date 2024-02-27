E = 6.88e4
nu = 0.25

[GlobalParams]
  large_kinematics = true
[]

[Variables]
  [disp_x]
  []
  [disp_y]
  []
  [disp_z]
  []
[]

[AuxVariables]
  [T]
    initial_value = 500
  []
[]

[Mesh]
  [msh]
    type = GeneratedMeshGenerator
    dim = 3
    nx = 1
    ny = 1
    nz = 1
  []
[]

[Kernels]
  [sdx]
    type = TotalLagrangianStressDivergence
    variable = disp_x
    component = 0
    displacements = 'disp_x disp_y disp_z'
  []
  [sdy]
    type = TotalLagrangianStressDivergence
    variable = disp_y
    component = 1
    displacements = 'disp_x disp_y disp_z'
  []
  [sdz]
    type = TotalLagrangianStressDivergence
    variable = disp_z
    component = 2
    displacements = 'disp_x disp_y disp_z'
  []
[]

[BCs]
  [fix_x]
    type = DirichletBC
    variable = disp_x
    boundary = 'left'
    value = 0.0
  []
  [fix_y]
    type = DirichletBC
    variable = disp_y
    boundary = 'bottom'
    value = 0.0
  []
  [fix_z]
    type = DirichletBC
    variable = disp_z
    boundary = 'back'
    value = 0.0
  []
  [pull_x]
    type = FunctionDirichletBC
    variable = disp_x
    boundary = 'right'
    function = 't'
    preset = true
  []
[]

[Materials]
  [elastic_tensor]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = ${units 70000 MPa}
    poissons_ratio = 0.33
  []
  [compute_strain]
    type = ComputeLagrangianStrain
    displacements = 'disp_x disp_y disp_z'
  []
  [flow_stress]
    type = DerivativeParsedMaterial
    property_name = flow_stress
    # sy   (MPa)
    # K    (MPa)
    # T... (deg C)
    # Pure Al (https://hal.science/hal-03434373/file/PIMM_EJM_2021_SEDDIK.pdf) 10.1016/j.euromechsol.2021.104432
    expression = 'sy:=90; K:=200; C:=0.035; n:=0.3; m:=1; ep_dot_0:=0.01;
                  ep_dot:=(ep-ep_old)/dt; T_r:=20; T_m:=660.3; T_star:=(T-T_r)/(T_m-T_r); ep_dot_star:=ep_dot/ep_dot_0;
                  (sy+K*ep^n)*(1+C*log(ep_dot_star))*(1-T_star^m)'
    material_property_names = 'ep:=effective_plastic_strain ep_old:=Old[effective_plastic_strain]'
    coupled_variables = T
    additional_derivative_symbols = 'ep'
    extra_symbols = dt
    derivative_order = 2
    compute = false
  []
  [compute_stress]
    type = ComputeSimoHughesJ2PlasticityStress
    flow_stress_material = flow_stress
  []
[]

[Executioner]
  type = Transient

  solve_type = NEWTON
  line_search = none

  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'

  nl_rel_tol = 1e-8
  nl_abs_tol = 1e-10

  start_time = 0.0
  dt = 5e-4
  num_steps = 20
[]
