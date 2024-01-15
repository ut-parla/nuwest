import argparse
import numpy as np
import sys

import pykokkos as pk
pk.set_default_space(pk.OpenMP)

from parla import Parla, TaskSpace, spawn
from parla.cython.device_manager import cpu, gpu
from parla.cython.tasks import AtomicTaskSpace as TaskSpace
from parla.tasks import get_current_context

from advection_kernel import advect

def main(in_N, in_steps):

    print("Initializing program to run on CPU\n")

    threads_per_block = 64
    num_blocks = 1024

    N = in_N
    num_steps = in_steps

    # Set up data structures on CPU

    x_ar = np.random.rand(N)
    v_ar = 0.01 * np.random.rand(N)
    E_ar = 0.01 * np.random.rand(N)
    R_ar = np.zeros(N)

    print("\nLaunching main Parla task")

    print("Beginning average position =", np.mean(x_ar))

    @spawn(placement=cpu)
    async def main_task():
        mytaskspace = TaskSpace("mytaskspace")
             
        for step in range(num_steps):
            # Task 0: particle kernel on CPU
            deps0 = [mytaskspace[1,step-1]] if step != 0 else []
            @spawn(mytaskspace[0,step], placement=cpu, dependencies=deps0)
            def gpu_task():
                # Draw random numbers with NumPy           
                R_ar[:] = np.random.rand(N)            

                # Pykokkos kernel for particle advection + collision
                advect(N, x_ar, v_ar, E_ar, R_ar, threads_per_block, num_blocks)

            # Task 1: generate random electric field on CPU
            deps1 = [mytaskspace[0,step]]
            @spawn(mytaskspace[1,step], placement=cpu, dependencies=deps1)
            def cpu_task():
                E_ar.fill(0.01 * np.random.rand())

        await mytaskspace
        mytaskspace.wait()
        print("End average position =", np.mean(x_ar))
    
    print("Complete, exiting.") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num_particles", type=int, default=100000)
    parser.add_argument("-s", "--num_steps", type=int, default=100)
    args = parser.parse_args()
    with Parla():
        main(args.num_particles, args.num_steps)
