# FFTSemiImplicit

!syntax description /TensorTimeIntegrator/FFTSemiImplicit

Semi-implicit time integrator that updates a field in Fourier space. Let `\bar u` denote the
Fourier transform of the real-space unknown `u`. With a linear spectral term `L(\vec k)` and a
nonlinear spectral term `N(\vec k)`, the first-order semi-implicit update writes

`\bar u^{n+1} = (\bar u^{n} + \Delta t\, N^{n}) / (1 - \Delta t\, L)`,

where `\Delta t` is the (sub-)time step. An optional second-order variant (one-step Adams-Bashforth
for `N`) is used when a single history state is available:

`\bar u^{n+1} = (\bar u^{n} + \tfrac{\Delta t}{2} (3 N^{n} - N^{n-1})) / (1 - \Delta t\, L)`.

Inputs are provided as reciprocal-space buffers:

- `reciprocal_buffer`: `\bar u^{n}`.
- `linear_reciprocal`: the linear prefactor `L(\vec k)`.
- `nonlinear_reciprocal`: the nonlinear term `N^{n}` (and its history if available).
- `history_size`: number of old states to store (controls order).

After assembling `\bar u^{n+1}`, this integrator computes the inverse FFT to update the real-space
buffer.

## Example Input File Syntax

!listing
[TensorTimeIntegrators]
  [c]
    type = FFTSemiImplicit
    buffer = c
    reciprocal_buffer = cbar             # ForwardFFT of c
    linear_reciprocal = kappabarbar      # L(k), e.g. kappa * |k|^2 or similar
    nonlinear_reciprocal = Mbarmubar     # N(c) in reciprocal space
    history_size = 1                     # enable 2nd-order nonlinear extrapolation
  []
[]
!listing-end

See also: [ForwardFFT](tensor_computes/PerformFFT.md) and
[FFTGradient](tensor_computes/FFTGradient.md).

!syntax parameters /TensorTimeIntegrator/FFTSemiImplicit

!syntax inputs /TensorTimeIntegrator/FFTSemiImplicit

!syntax children /TensorTimeIntegrator/FFTSemiImplicit

