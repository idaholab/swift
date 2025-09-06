# HyperElasticIsotropic

!syntax description /TensorComputes/Solve/HyperElasticIsotropic

HyperElasticIsotropic implements a small set of hyperelastic isotropic relations. Given deformation gradient $\mathbf{F}$, shear modulus $\mu$, and bulk modulus $K$, it computes stress and an optional consistent tangent operator.

## Overview

Inputs:
- [!param](/TensorComputes/Solve/HyperElasticIsotropic/F): deformation gradient tensor $\mathbf{F}$ (rank\-2).
- [!param](/TensorComputes/Solve/HyperElasticIsotropic/mu): shear modulus $\mu$.
- [!param](/TensorComputes/Solve/HyperElasticIsotropic/K): bulk modulus $K$.

Outputs:
- Primary output: first Piola\-Kirchhoff\-like stress $\mathbf{P}$ computed from $\mathbf{F}$ and isotropic 4th\-order moduli.
- Optional: [!param](/TensorComputes/Solve/HyperElasticIsotropic/tangent_operator) to receive the 4th\-order stiffness.

## Example Input File Syntax

!listing test/tests/mechanics/mech.i block=TensorComputes/Solve/hyper_elasticity

!syntax parameters /TensorComputes/Solve/HyperElasticIsotropic

!syntax inputs /TensorComputes/Solve/HyperElasticIsotropic

!syntax children /TensorComputes/Solve/HyperElasticIsotropic

