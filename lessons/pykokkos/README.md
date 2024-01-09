## PyKokkos Tutorial

In this tutorial, we will showcase the most important/exciting
features of PyKokkos.  Further details, examples, and documentation
can be found on the [PyKokkos web
page](https://github.com/kokkos/pykokkos).


### Getting Started

Although you can set up PyKokkos to run natively on your machine, we
recommend using our Docker images for this tutorial, which will save
you the installation time.

TODO: depending if we are going to use pykokkos or parla+pykokkos images


### Features

We will cover the following features of PyKokkos:

* workunits, i.e., kernels ([example](/scripts/mini_boltzmann_cpu.py))
* parallel processing, i.e., `parallel_for` ([example](/scripts/mini_boltzmann_cpu.py))
* interoperability with `numpy` and `cupy` ([example](/scripts/mini_boltzmann_cpu.py))
* CPU runs ([example](/scripts/mini_boltzmann_cpu.py))
* GPU runs ([example](/scripts/mini_boltzmann_gpu.py))
* multi-GPU runs ([example](/scripts/mini_boltzmann_multigpu.py))
* atomics
* scratch memory
* ufuncs
* hierarchical parallelism
* C++ code generation and bindings
* profiling

### Examples

We will be covering the aforementioned features using two examples:
simplified Boltzmann solver and stencil computation.  The latter one
will be an exercise for the audience.

#### Boltzmann Simplified

One of the examples we will be using in this tutorial is a stripped
down version of the Boltzmann solver.  (We also provide a more
complete version of the solver in this repository for completeness,
but we do not discuss the complex version.)

TODO: short explanation of the problem

#### Stencil

TODO: short explanation of the problem
