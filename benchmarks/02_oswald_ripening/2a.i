[Domain]
  dim = 2
  nx = 80
  ny = 80
  xmax = 200
  ymax = 200

  device_names = 'cuda'

  mesh_mode = DOMAIN
[]

fchem='fa:=rho^2*(c-ca)^2;
fb:=rho^2*(cb-c)^2;
h:=n1^3*(6*n1^2-15*n1+10)+n2^3*(6*n2^2-15*n2+10)+n3^3*(6*n3^2-15*n3+10)+n4^3*(6*n4^2-15*n4+10);
g:=n1^2*(1-n1)^2+n2^2*(1-n2)^2+n3^2*(1-n3)^2+n4^2*(1-n4)^2+
alpha*(
n1^2*n2^2 + n1^2*n3^2 + n1^2*n4^2 +
n2^2*n1^2 + n2^2*n3^2 + n2^2*n4^2 +
n3^2*n1^2 + n3^2*n2^2 + n3^2*n4^2 +
n4^2*n1^2 + n4^2*n2^2 + n4^2*n3^2);
(fa*(1-h) + fb*h + w*g)'

cnames = 'rho     ca  cb  alpha w L M'
cvalues= 'sqrt(2) 0.3 0.7 5     1 5 5'

[TensorBuffers]
  # variables
  [c]
    # map_to_aux_variable = c
  []
  [n1]
  []
  [n2]
  []
  [n3]
  []
  [n4]
  []

  [c_bar]
  []
  [n1_bar]
  []
  [n2_bar]
  []
  [n3_bar]
  []
  [n4_bar]
  []

  [mu_c]
    # map_to_aux_variable = mu
  []
  [mu_n1]
  []
  [mu_n2]
  []
  [mu_n3]
  []
  [mu_n4]
  []

  [mu_c_bar]
  []

  [Mbar_mu_c_bar]
  []

  # constant tensors
  [Lbar] # FFT(M*laplacian)
  []
  [MkappaL2bar] # FFT(-M*kappa*laplacian^2)
  []
  [kappaLbar] # FFT(L*kappa*laplacian)
  []

  # postprocessing
  [F]
  []
  [Fgrad]
  []
  [bnds]
  []
[]

[TensorComputes]
  [Initialize]
    [c]
      type = ParsedCompute
      buffer = c
      extra_symbols = true
      expression = 'c0+epsilon*(cos(0.105*x)*cos(0.11*y)+(cos(0.13*x)*cos(0.087*y))^2+cos(0.025*x-0.15*y)*cos(0.07*x-0.02*y))'
      constant_names = 'c0 epsilon'
      constant_expressions = '0.5 0.01'
    []
    [Lbar]
      type = ReciprocalLaplacianFactor
      # Mobility is pulled into the chemical potential below
      buffer = Lbar
    []
    [MkappaL2bar]
      type = ReciprocalLaplacianSquareFactor
      factor = -15 # -kappa_c*M
      buffer = MkappaL2bar
    []
    [kappaLbar]
      type = ReciprocalLaplacianFactor
      buffer = kappaLbar
      factor = 15 # kappa_ni*L
    []
  []

  [Solve]
    [mu_c]
      type = ParsedCompute
      buffer = mu_c
      enable_jit = true
      expression = '${fchem}*M'
      constant_names = ${cnames}
      constant_expressions = ${cvalues}
      derivatives = c
      inputs = 'c n1 n2 n3 n4'
    []

    [mu_n1]
      type = ParsedCompute
      buffer = mu_n1
      enable_jit = true
      expression = '${fchem}*L'
      constant_names = ${cnames}
      constant_expressions = ${cvalues}
      derivatives = n1
      inputs = 'c n1 n2 n3 n4'
    []
    [mu_n2]
      type = ParsedCompute
      buffer = mu_n2
      enable_jit = true
      expression = '${fchem}*L'
      constant_names = ${cnames}
      constant_expressions = ${cvalues}
      derivatives = n2
      inputs = 'c n1 n2 n3 n4'
    []
    [mu_n3]
      type = ParsedCompute
      buffer = mu_n3
      enable_jit = true
      expression = '${fchem}*L'
      constant_names = ${cnames}
      constant_expressions = ${cvalues}
      derivatives = n3
      inputs = 'c n1 n2 n3 n4'
    []
    [mu_n4]
      type = ParsedCompute
      buffer = mu_n4
      enable_jit = true
      expression = '${fchem}*L'
      constant_names = ${cnames}
      constant_expressions = ${cvalues}
      derivatives = n4
      inputs = 'c n1 n2 n3 n4'
    []

    [mu_c_bar]
      type = ForwardFFT
      buffer = mu_c_bar
      input = mu_c
    []
    [mu_n1_bar]
      type = ForwardFFT
      buffer = mu_n1_bar
      input = mu_n1
    []
    [mu_n2_bar]
      type = ForwardFFT
      buffer = mu_n2_bar
      input = mu_n2
    []
    [mu_n3_bar]
      type = ForwardFFT
      buffer = mu_n3_bar
      input = mu_n3
    []
    [mu_n4_bar]
      type = ForwardFFT
      buffer = mu_n4_bar
      input = mu_n4
    []

    [Mbar_mu_c_bar]
      type = ParsedCompute
      buffer = Mbar_mu_c_bar
      enable_jit = true
      expression = 'Lbar*mubar'
      inputs = 'Lbar mubar'
    []

    [c_bar]
      type = ForwardFFT
      buffer = c_bar
      input = c
    []
    [n1_bar]
      type = ForwardFFT
      buffer = n1_bar
      input = n1
    []
    [n2_bar]
      type = ForwardFFT
      buffer = n2_bar
      input = n2
    []
    [n3_bar]
      type = ForwardFFT
      buffer = n3_bar
      input = n3
    []
    [n4_bar]
      type = ForwardFFT
      buffer = n4_bar
      input = n4
    []
  []

  [Postprocess]
    [Fgrad]
      type = FFTGradientSquare
      buffer = Fgrad
      input = c
      factor = 1 # kappa/2
    []
    [F]
      type = ParsedCompute
      buffer = F
      enable_jit = true
      expression = ${fchem}
      constant_names =  ${cnames}
      constant_expressions = ${cvalues}
      inputs = 'c n1 n2 n3 n4'
    []
    [bnds]
      type = ParsedCompute
      buffer = bnds
      enable_jit = true
      expression = '(n1*n2)+(n1*n3)+(n1*n4)+(n2*n3)+(n2*n4)+(n3*n4)'
    []
  []
[]

[TensorTimeIntegrators]
  # the linear operator in this problem is diagonal, which simplifies time integration considerably!
  [c]
    type = FFTSemiImplicit
    buffer = c
    reciprocal_buffer = c_bar
    linear_reciprocal = MkappaL2bar
    nonlinear_reciprocal = Mbar_mu_c_bar
  []
  [n1]
    type = FFTSemiImplicit
    buffer = n1
    reciprocal_buffer = n1_bar
    linear_reciprocal = kappaLbar
    nonlinear_reciprocal = mu_n1_bar
  []
  [n2]
    type = FFTSemiImplicit
    buffer = n2
    reciprocal_buffer = n2_bar
    linear_reciprocal = kappaLbar
    nonlinear_reciprocal = mu_n2_bar
  []
  [n3]
    type = FFTSemiImplicit
    buffer = n3
    reciprocal_buffer = n3_bar
    linear_reciprocal = kappaLbar
    nonlinear_reciprocal = mu_n3_bar
  []
  [n4]
    type = FFTSemiImplicit
    buffer = n4
    reciprocal_buffer = n4_bar
    linear_reciprocal = kappaLbar
    nonlinear_reciprocal = mu_n4_bar
  []
[]

[AuxVariables]
  # [c]
  #   # family = MONOMIAL
  #   # order = CONSTANT
  # []
[]

[Postprocessors]
  [min_c]
    type = TensorExtremeValuePostprocessor
    buffer = c
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_c]
    type = TensorExtremeValuePostprocessor
    buffer = c
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
  [F]
    type = TensorIntegralPostprocessor
    buffer = F
  []
  # [stable_dt]
  #   type = SemiImplicitCriticalTimeStep
  #   buffer = MkappaL2bar
  # []
[]

[Problem]
  type = TensorProblem
  spectral_solve_substeps = 1000
[]

[Executioner]
  type = Transient
  num_steps = 400
  [TimeStepper]
    type = IterationAdaptiveDT
    growth_factor = 1.1
    dt = 1
  []
  dtmax = 300
[]

[Outputs]
  # exodus = true
  csv = true
  perf_graph = true
  execute_on = 'TIMESTEP_END'
[]
