# ReciprocalIntegral

!syntax description /Postprocessors/ReciprocalIntegral

This postprocessor acts on a reciprocal\-space tensor and extracts the magnitude of the zero wave
vector (constant contribution). The result matches the
[TensorIntegralPostprocessor](TensorIntegralPostprocessor.md) applied to the corresponding real\-space field.

## Example Input File Syntax

!listing test/tests/postprocessors/postprocessors.i block=Postprocessors/int_c_bar

!syntax parameters /Postprocessors/ReciprocalIntegral

!syntax inputs /Postprocessors/ReciprocalIntegral

!syntax children /Postprocessors/ReciprocalIntegral
