[Domain]
  dim = 2
  nx = 20
  ny = 20
  xmax = ${fparse pi*4}
  ymax = ${fparse pi*4}
  device_names = 'cpu'
  mesh_mode = DUMMY
[]

[TensorBuffers]
  [eta]
  []
  [eta_bar]
  []
  [eta2]
  []
  [zero]
  []
[]

[TensorComputes]
  [Initialize]
    [eta]
      type = ParsedCompute
      buffer = eta
      expression = 'sin(x)+sin(y)+sin(z)'
      extra_symbols = true
    []
    [eta2]
      type = ConstantTensor
      buffer = eta2
      real = 1
    []
    [zero]
      type = ConstantReciprocalTensor
      buffer = zero
      real = 0
      imaginary = 0
    []
  []

  [Solve]
    [eta_bar]
      type = ForwardFFT
      buffer = eta_bar
      input = eta
    []
    [eta_2]
      type = InverseFFT
      buffer = eta2
      input = eta_bar
    []
  []
[]

[TensorSolver]
  type = SemiImplicitSolver
  buffer = eta
  reciprocal_buffer = eta_bar
  linear_reciprocal = zero
  nonlinear_reciprocal = zero
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 2
[]

[TensorOutputs]
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'eta eta2'
    enable_hdf5 = true
  []
[]
