#### Boundary ####
[Boundary]
  ##### for fluid
  [top]
    type = LBMBounceBack
    buffer = f
    f_old = fpc
    boundary = top
  []

  [bottom]
    type = LBMBounceBack
    buffer = f
    f_old = fpc
    boundary = bottom
  []

  ##### for temperature
  [temperature_g_top]
    type = LBMFixedZerothOrderBC9Q
    buffer = g
    f = g
    value = T_C
    boundary = top
  []

  [temperature_g_bottom]
    type = LBMFixedZerothOrderBC9Q
    buffer = g
    f = g
    value = T_H
    boundary = bottom
  []
[]
