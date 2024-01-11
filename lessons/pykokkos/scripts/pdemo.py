import numpy as np
import sys


from parla import Parla, TaskSpace, spawn
from parla.cython.device_manager import cpu, gpu
from parla.cython.tasks import AtomicTaskSpace as TaskSpace
from parla.tasks import get_current_context

import pykokkos as pk

pk.set_default_space(pk.OpenMP)

from advection_kernel import advect


def main():
    threads_per_block = 64
    num_blocks = 1024

    N = 10000
    num_steps = 10

    x_ar = np.random.rand(N)
    v_ar = 0.01 * np.random.rand(N)
    E_ar = np.zeros(N)

    d_x_ar = pk.array(x_ar)
    d_v_ar = pk.array(v_ar)
    d_E_ar = pk.array(E_ar)

    print("\n Launching main Parla task to run for", num_steps, "steps\n")

    @spawn(placement=cpu)
    def main_task():
        mytaskspace = TaskSpace("mytaskspace")

        for step in range(num_steps):
            # Task 0: Advection
            deps0 = [mytaskspace[0, step - 1]] if step != 0 else []

            @spawn(mytaskspace[0, step], placement=cpu, dependencies=deps0)
            def task0():
                # Advect particles
                advect(N, d_x_ar, d_v_ar, d_E_ar, threads_per_block, num_blocks, step)

            ## Task 1: Electric Field
            # deps1 = [mytaskspace[0,step]]
            # @spawn(mytaskspace[1,step], placement=cpu, dependencies=deps1)
            # def task1():
            #    E_ar.fill(0.01 * np.random.rand())
        mytaskspace.wait()
        print("Complete, exiting.")


if __name__ == "__main__":
    with Parla():
        main()
