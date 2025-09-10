# ParsedCompute

!syntax description /TensorComputes/Solve/ParsedCompute

## Overview

Evaluates a user\-provided algebraic expression to compute the target buffer from named inputs and
coordinates. Specify the expression via
[!param](/TensorComputes/Solve/ParsedCompute/expression) and list input buffers with
[!param](/TensorComputes/Solve/ParsedCompute/inputs). Enabling
[!param](/TensorComputes/Solve/ParsedCompute/extra_symbols) adds symbols like `x`, `y`, `z`.

## Example Input File Syntax

!listing test/tests/gradient/gradient_square.i block=TensorComputes/Initialize/diff

!syntax parameters /TensorComputes/Solve/ParsedCompute

!syntax inputs /TensorComputes/Solve/ParsedCompute

!syntax children /TensorComputes/Solve/ParsedCompute
