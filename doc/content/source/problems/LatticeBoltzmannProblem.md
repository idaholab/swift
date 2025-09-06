# LatticeBoltzmannProblem

!syntax description /Problem/LatticeBoltzmannProblem

The problem object derived from TensorProblem (/TensorProblem.md) is used to solve Lattice Boltzmann simulations. It works in the same way as TensorProblem, with a few extra features. It adds stencil and boundary condition object and modifies the initialization and execution order in favor of LBM.

## Overview

Problem driver specialized for Lattice Boltzmann simulations: manages stencils, distribution buffers,
macroscopic fields, and substepping between collision and streaming. Provides access to LBM
constants like `c_s` and time step for objects that require them.

## Example Input File Syntax

!listing test/tests/lbm/channel2D.i block=Problem

!syntax parameters /Problem/LatticeBoltzmannProblem

!syntax inputs /Problem/LatticeBoltzmannProblem

!syntax children /Problem/LatticeBoltzmannProblem
