# LBMSmagorinskyMRTCollision

!syntax description /TensorComputes/Solve/LBMSmagorinskyMRTCollision

This objects implements commonly used collision dynamics for Lattice Boltzmann simulation. Currently available collision operators are BGK single relaxation time and Multi Relaxation Time (MRT) operators. The additional operation is available to project the non-equilibrium distribution onto Hermite space to achieve stability with the boolean parameter `projection`. For high Reynolds number simulations, Smagorinsky LES collision dynamics is available for both BGK and MRT operators.

## Overview

!! Replace these lines with information regarding the LBMSmagorinskyMRTCollision object.

## Example Input File Syntax

!! Describe and include an example of how to use the LBMSmagorinskyMRTCollision object.

!syntax parameters /TensorComputes/Solve/LBMSmagorinskyMRTCollision

!syntax inputs /TensorComputes/Solve/LBMSmagorinskyMRTCollision

!syntax children /TensorComputes/Solve/LBMSmagorinskyMRTCollision
