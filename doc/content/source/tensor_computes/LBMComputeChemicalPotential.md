# LBMComputeChemicalPotential

!syntax description /TensorComputes/Solve/LBMComputeChemicalPotential

This compute object computes chemical potential from parabiolic free energy eqution for lattice Boltzmann simulations.

## Overview

Evaluates the Cahn\-Hilliard chemical potential for a phase field $\phi$ using a double\-well
potential and interfacial energy term. Provide the scalar field via
[!param](/TensorComputes/Solve/LBMComputeChemicalPotential/phi) and its Laplacian via
[!param](/TensorComputes/Solve/LBMComputeChemicalPotential/laplacian_phi). Control the interface
thickness with [!param](/TensorComputes/Solve/LBMComputeChemicalPotential/thickness) and the
surface tension with [!param](/TensorComputes/Solve/LBMComputeChemicalPotential/sigma).

## Example Input File Syntax

!listing test/tests/lbm/phase.i block=TensorComputes/Solve/potential

!syntax parameters /TensorComputes/Solve/LBMComputeChemicalPotential

!syntax inputs /TensorComputes/Solve/LBMComputeChemicalPotential

!syntax children /TensorComputes/Solve/LBMComputeChemicalPotential
