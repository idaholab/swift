# SemiImplicitCriticalTimeStep

!syntax description /UserObjects/SemiImplicitCriticalTimeStep

Compute the critical time step for teh semi implicit time integration, based on the

## Overview

Estimates a stable time step for semi\-implicit integration based on a linearized operator in
reciprocal space. Provide the reciprocal buffer via
[!param](/Postprocessors/SemiImplicitCriticalTimeStep/buffer).

## Example Input File Syntax

!listing test/tests/cahnhilliard/cahnhilliard.i block=Postprocessors/min_c

!syntax parameters /UserObjects/SemiImplicitCriticalTimeStep

!syntax inputs /UserObjects/SemiImplicitCriticalTimeStep

!syntax children /UserObjects/SemiImplicitCriticalTimeStep
