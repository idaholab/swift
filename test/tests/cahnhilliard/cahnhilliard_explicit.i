#
# Simple Cahn-Hilliard solve on a 2D grid. We create a matching (conforming)
# MOOSE mesh (with one element per FFT grid cell) and project the solution onto
# the MOOSE mesh to utilize the exodus output object.
#

[Domain]
    dim = 2
    nx = 50
    ny = 50
    xmax = 3
    ymax = 3
    mesh_mode = DOMAIN
    device_names = cpu
[]

[TensorBuffers]
    [c]
        map_to_aux_variable = c
    []
    [cbar]
    []
    [mu]
        map_to_aux_variable = mu
    []
    [mubar]
    []
    [dc_dt_bar]
    []
    # constant tensors
    [Mbar]
    []
    [Mkappabarbar]
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
        [Mkappabarbar]
            type = ReciprocalLaplacianSquareFactor
            factor = '${fparse 0.2 * 1e-4}' # M * kappa
            buffer = Mkappabarbar
        []
        [dc_dt_bar_IC]
            type = ConstantReciprocalTensor
            buffer = dc_dt_bar
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
        [dc_dt_bar]
            type = ParsedCompute
            buffer = dc_dt_bar
            enable_jit = true
            expression = 'Mbar*mubar - Mkappabarbar*cbar'
            inputs = 'Mbar mubar Mkappabarbar cbar'
            real_space = false
        []
        [cbar]
            type = ForwardFFT
            buffer = cbar
            input = c
        []

        # root compute
        [cahn_hilliard]
            type = ComputeGroup
            computes = 'mu mubar dc_dt_bar cbar'
        []
    []
[]

[TensorSolver]
    type = ForwardEulerSolver
    time_derivative_reciprocal = dc_dt_bar
    root_compute = cahn_hilliard
    buffer = c
    reciprocal_buffer = cbar
    substeps = 50
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

[Problem]
    type = TensorProblem
[]

[Executioner]
    type = Transient
    num_steps = 100
    dt = 1e-1
[]

[Outputs]
    exodus = true
    csv = true
[]
