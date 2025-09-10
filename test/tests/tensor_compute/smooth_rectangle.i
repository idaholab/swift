[Domain]
    dim = 2
    nx = 100
    ny = 100
    xmax = 20
    ymax = 20
    mesh_mode = DUMMY
    device_names = cpu
[]

[TensorComputes]
    [Initialize]
        [rectangle_sharp]
            type = SmoothRectangleCompute
            buffer = rectangle_sharp
            x1 = 5
            x2 = 15
            y1 = 5
            y2 = 15
            inside = -1
            outside = 3
        []
        [rectangle_cos]
            type = SmoothRectangleCompute
            buffer = rectangle_cos
            x1 = 5
            x2 = 15
            y1 = 5
            y2 = 15
            inside = -1
            outside = 3
            profile = COS
            int_width = 1
        []
        [rectangle_tanh]
            type = SmoothRectangleCompute
            buffer = rectangle_tanh
            x1 = 5
            x2 = 15
            y1 = 5
            y2 = 15
            inside = -1
            outside = 3
            profile = TANH
            int_width = 1
        []
    []
[]

[Problem]
    type = TensorProblem
[]

[TensorOutputs]
    [xdmf]
        type = XDMFTensorOutput
        buffer = 'rectangle_sharp rectangle_cos rectangle_tanh'
        enable_hdf5 = true
    []
[]

[Executioner]
    type = Transient
    num_steps = 0
[]

[Outputs]
    perf_graph = true
[]
