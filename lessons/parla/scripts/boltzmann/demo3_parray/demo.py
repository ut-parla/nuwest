import argparse
import numpy as np
import cupy as cp
from numba import cuda
import sys

from parla.array import asarray
from parla import Parla, TaskSpace, spawn
from parla.cython.device_manager import cpu, gpu
from parla.cython.tasks import AtomicTaskSpace as TaskSpace
from parla.tasks import get_current_context

import pykokkos as pk
pk.set_default_space(pk.Cuda)

from advection_kernel import advect

def main(in_gpus, in_N, in_steps):

    NUM_GPUS = int(in_gpus)

    print("Initializing program to run on", NUM_GPUS, " GPU(s)\n")

    threads_per_block = 64
    num_blocks = 1024

    N = 100000
    num_steps = 100

    # Set up data structures on each GPU
    p_x_ar_list = []
    p_v_ar_list = []
    p_R_ar_list = []
    p_E_ar_list = []

    # Set up data structures on each GPU
    for ng in range(NUM_GPUS):
        print("Setting up data structures corresponding to GPU", ng)
        
        temp_x_ar = np.random.rand(N)
        temp_v_ar = 0.01*np.random.rand(N)
        temp_R_ar = np.random.rand(N)
        temp_E_ar = np.zeros(N)

        p_temp_x_ar = asarray(temp_x_ar)
        p_temp_v_ar = asarray(temp_v_ar)
        p_temp_R_ar = asarray(temp_R_ar)
        p_temp_E_ar = asarray(temp_E_ar)

        p_x_ar_list.append(p_temp_x_ar)
        p_v_ar_list.append(p_temp_v_ar)
        p_R_ar_list.append(p_temp_R_ar)
        p_E_ar_list.append(p_temp_E_ar)

    print("\nLaunching main Parla task")
    
    for ng in range(NUM_GPUS):
        print("Beginning average position on GPU", ng, "=", np.mean(p_x_ar_list[ng].array))

    @spawn(placement=cpu)
    async def main_task():
        mytaskspace = TaskSpace("mytaskspace")
             
        for step in range(num_steps):

            # Task 0: particle kernel on each GPU
            for ng in range(NUM_GPUS):
                deps0 = [mytaskspace[1,step-1,0]] if step != 0 else []
                @spawn(mytaskspace[0,step,ng], placement=gpu(ng), dependencies=deps0, input=[p_x_ar_list[ng], p_v_ar_list[ng], p_R_ar_list[ng], p_E_ar_list[ng]])
                def gpu_task():
                    cp.cuda.Device(ng).use()
                    pk.set_device_id(ng)
    
                    p_R_ar_list[ng][:] = cp.random.rand(N)
 
                    # Advect particles
                    advect(N, p_x_ar_list[ng].array, p_v_ar_list[ng].array, p_E_ar_list[ng].array, p_R_ar_list[ng].array, threads_per_block, num_blocks)

            # Task 1: generate random electric field and generate random numbers on CPU
            deps1 = [mytaskspace[0,step,ng] for ng in range(NUM_GPUS)] 
            @spawn(mytaskspace[1,step,0], placement=cpu, dependencies=deps1, input=[p_E_ar_list[ng], p_R_ar_list[ng]])
            def cpu_task():
                
                # Drawn random electric field values
                p_E_ar_list[ng].array.fill(0.01 * np.random.rand())
               
                # Draw random numbers with NumPy 
                p_R_ar_list[ng].array[:] = np.random.rand(N)

        mytaskspace.wait()
        cp.cuda.get_current_stream().synchronize() 
        for ng in range(NUM_GPUS):
            print("End average position on GPU", ng, "=", np.mean(p_x_ar_list[ng].array))

    print("\nComplete, exiting.\n") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--num_gpus", type=int, default=1)
    parser.add_argument("-N", "--num_particles", type=int, default=100000)
    parser.add_argument("-s", "--num_steps", type=int, default=100)
    args = parser.parse_args()
    with Parla():
        main(args.num_gpus, args.num_particles, args.num_steps)
