[Mesh]
  type = FFTMesh
  dim = 2
  nx = 50
  ny = 50
  xmax = ${fparse pi*4}
  ymax = ${fparse pi*4}
[]

[FFTBuffers]
  [eta]
  []
[]

[FFTICs]
  [eta]
    type = SineIC
    buffer = eta
  []
[]

[AuxVariables]
  [eta]
  []
[]

[AuxKernels]
  [eta]
    type = FFTBufferAux
    buffer = eta
    variable = eta
    execute_on = TIMESTEP_END
  []
[]

[Problem]
  type = FFTProblem
[]

[Executioner]
  type = Transient
  num_steps = 2
[]

[Outputs]
  exodus = true
[]
