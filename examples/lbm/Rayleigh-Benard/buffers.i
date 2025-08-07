[TensorBuffers]
  # Density distribution functions
  [f]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [feq]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [fpc]
    type = LBMTensorBuffer
    buffer_type = df
  []
  # Temperature distribution functions
  [g]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [geq]
    type = LBMTensorBuffer
    buffer_type = df
  []
  [gpc]
    type = LBMTensorBuffer
    buffer_type = df
  []
  # Fluid macroscopic variables: density and velocity
  [density]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  [velocity]
    type = LBMTensorBuffer
    buffer_type = mv
  []
  # Temperature macroscpic variables: temperature and 'velocity'
  [T]
    type = LBMTensorBuffer
    buffer_type = ms
  []
  # Forces
  [F]
    type = LBMTensorBuffer
    buffer_type = mv
  []
[]
