# AdamsBashforthMoulton

!syntax description /TensorSolver/AdamsBashforthMoulton

Semi\-implicit time integrator using Adams\-Bashforth prediction and Adams\-Moulton correction. The
predictor and corrector orders, as well as the number of corrector iterations, are configurable via
[!param](/TensorSolver/AdamsBashforthMoulton/predictor_order),
[!param](/TensorSolver/AdamsBashforthMoulton/corrector_order), and
[!param](/TensorSolver/AdamsBashforthMoulton/corrector_steps). Subcycling is controlled by
[!param](/TensorSolver/AdamsBashforthMoulton/substeps).

## Example Input File Syntax

!listing test/tests/tensor_compute/group.i block=TensorSolver

!syntax parameters /TensorSolver/AdamsBashforthMoulton

!syntax inputs /TensorSolver/AdamsBashforthMoulton

!syntax children /TensorSolver/AdamsBashforthMoulton
