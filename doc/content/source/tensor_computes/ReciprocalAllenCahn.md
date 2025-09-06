# ReciprocalAllenCahn

!syntax description /TensorComputes/Solve/ReciprocalAllenCahn

ReciprocalAllenCahn computes the Allen\-Cahn bulk driving term masked by `psi` and returns the result in reciprocal space. The rate is $\dot{\eta} = - L \, \partial F_{chem} / \partial \eta$ inside `psi>0` and zero elsewhere, then transformed by FFT.

## Overview

Inputs:
- `dF_chem_deta`: chemical potential derivative $\partial F_{chem} / \partial \eta$.
- `L`: Allen\-Cahn mobility.
- `psi`: mask field; only positive regions evolve. Set `always_update_psi = true` if the mask changes over time.

Output is the Fourier transform of the masked rate.

## Example Input File Syntax

!listing test/tests/kks/KKS_no_flux_bc.i block=TensorComputes/Solve/AC_bulk

!syntax parameters /TensorComputes/Solve/ReciprocalAllenCahn

!syntax inputs /TensorComputes/Solve/ReciprocalAllenCahn

!syntax children /TensorComputes/Solve/ReciprocalAllenCahn

