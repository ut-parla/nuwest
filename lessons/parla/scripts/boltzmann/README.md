## Sample Advection Kernels

This directory contains a set of sample advection kernels that can be used to show Parla execution. 

In this tutorial we will be using a stripped down version of a particle Boltzmann solver. (We also provide a more complete version of the solver in this repository for completeness,
but we do not discuss the complex version.)

We model a system of particles in a 1-D domain. 

At each timestep:
-  a random electric field is drawn
- particles undergo advection
in physical space
- advection in velocity space
- possible a random reflective collision (which flips the direction of their velocity), and
reflective collision with the boundary walls.

