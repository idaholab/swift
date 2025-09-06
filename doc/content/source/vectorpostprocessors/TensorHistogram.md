# TensorHistogram

!syntax description /VectorPostprocessors/TensorHistogram

This VectorPostprocessor will compute a histogram of the provided tensor with user defined lower and upper bounds and a given number of bins.
Two columns will be generated. `bin` contains the center value of the histogram bin corresponding to the vector row, and `count` contains the
number of cells with a value that falls into the histogram bin.

## Example Input File Syntax

!listing test/tests/histogram/test.i block=VectorPostprocessors/hist

!syntax parameters /VectorPostprocessors/TensorHistogram

!syntax inputs /VectorPostprocessors/TensorHistogram

!syntax children /VectorPostprocessors/TensorHistogram
