[Domain]
  dim = 3
  nx = 5
  ny = 5
  nz = 5
  mesh_mode = DUMMY
[]

[Stencil]
  [d3q19]
    type = LBMD3Q19
  []
[]

[TensorBuffers]
  [density]
    type=LBMTensorBuffer
    buffer_type = ms
  []
[]

[TensorComputes]
  [Initialize]
    [density]
      type = LBMConstantTensor
      buffer = density
      constants = 1.0
    []
  []
  [Boundary]
    [left]
      type = LBMMacroscopicNeumannBC
      buffer = density
      value = 0.1
      boundary = left
    []
    [right]
      type = LBMMacroscopicNeumannBC
      buffer = density
      value = 0.1
      boundary = right
    []
    [top]
      type = LBMMacroscopicNeumannBC
      buffer = density
      value = 0.1
      boundary = top
    []
    [bottom]
      type = LBMMacroscopicNeumannBC
      buffer = density
      value = 0.1
      boundary = bottom
    []
    [front]
      type = LBMMacroscopicNeumannBC
      buffer = density
      value = 0.1
      boundary = front
    []
    [back]
      type = LBMMacroscopicNeumannBC
      buffer = density
      value = 0.1
      boundary = back
    []
  []
[]

[Problem]
  type = LatticeBoltzmannProblem
  substeps = 1
[]

[Executioner]
  type = Transient
  num_steps = 2
[]

[TensorOutputs]
  [xdmf2]
    type = XDMFTensorOutput
    buffer = 'density'
    output_mode = 'Cell'
    enable_hdf5 = true
  []
[]
