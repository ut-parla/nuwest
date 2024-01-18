## PyKokkos Tutorial

In this tutorial, we will showcase the most important/exciting
features of PyKokkos.  Further details, examples, and documentation
can be found on the [PyKokkos web
page](https://github.com/kokkos/pykokkos).

### Getting Started

Although you can set up PyKokkos to run natively on your machine, we
recommend using our Docker images for this tutorial, which will save
you the installation time.

You can setup the environment by running the following command:
```
./s
```

Once that is complete, you will be able to run provided examples like so:
```
./s scripts/mini_boltzmann_cpu.py -N 100000 -s 10
```

### Examples

We will be covering the aforementioned features using two examples:
simplified Boltzmann solver and stencil computation.  The latter one
will be an exercise for the audience.

#### Boltzmann Simplified

One of the examples we will be using in this tutorial is a stripped
down version of a particle Boltzmann solver. (We also provide a more
complete version of the solver in this repository for completeness,
but we do not discuss the complex version.)

We model a system of particles in a 1-D domain. At each timestep, a
random electric field is drawn, and then particles undergo advection
in physical space, advection in velocity space, random reflective
collision (which flips the direction of their velocity), and
reflective collision with the boundary walls.

#### Stencil

Another example we will be illustrating using PyKokkos is a stencil
code. For a fixed number of iterations, we apply a stencil kernel
followed by a gemv and a dot product operation.

### Features

We will cover the following features of PyKokkos:

* workunits, i.e., kernels ([example](/lessons/pykokkos/scripts/mini_boltzmann_cpu.py))
* parallel processing, i.e., `parallel_for` ([example](/lessons/pykokkos/scripts/mini_boltzmann_cpu.py))
* interoperability with `numpy` and `cupy` ([example](/lessons/pykokkos/scripts/mini_boltzmann_cpu.py))
* CPU runs ([example](/lessons/pykokkos/scripts/mini_boltzmann_cpu.py)) `./s scripts/mini_boltzmann_cpu.py -N 100000 -s 10`
* GPU runs ([example](/lessons/pykokkos/scripts/mini_boltzmann_gpu.py)) `./s scripts/mini_boltzmann_gpu.py -N 100000 -s 10`
* multi-GPU runs ([example](/lessons/pykokkos/scripts/mini_boltzmann_multigpu.py))
* atomics ([example](/lessons/pykokkos/scripts/mini_boltzmann_atomics.py)) `./s scripts/mini_boltzmann_atomics.py -N 100000 -s 10`
* ufuncs ([example](/lessons/pykokkos/scripts/cholesky.py)) `./s scripts/cholesky.py -N 100000 -s 10`
* hierarchical parallelism ([example](/lessons/pykokkos/scripts/mini_boltzmann_hp.py)) `./s scripts/mini_boltzmann_hp.py -N 100000 -s 10`
* scratch memory (optional)
* C++ code generation and bindings
* profiling `./s scripts/mini_boltzmann_profile.py -N 10 -s 100`
