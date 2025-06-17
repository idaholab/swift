[Settings]
  require_double_precision = false
[]

[Models]
  [multiply]
    type = ScalarMultiplication
    from_var = 'forces/A forces/B'
    to_var = 'state/C'
  []
[]

