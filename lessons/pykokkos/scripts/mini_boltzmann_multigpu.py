import argparse
import numpy as np
import cupy as cp
from numba import cuda
import sys
import pykokkos as pk
pk.set_default_space(pk.Cuda)

from advection_kernel import advect


def main(in_gpus, in_particles, in_steps):
    NUM_GPUS = in_gpus

    print("Initializing program to run on", NUM_GPUS, " GPU(s)\n")

    threads_per_block = 64
    num_blocks = 1024

    N = in_particles
    num_steps = in_steps

    x_ar_list = []
    v_ar_list = []
    gpu_E_ar_list = []
    gpu_R_ar_list = []

    d_x_ar_list = []
    d_v_ar_list = []
    d_E_ar_list = []
    d_R_ar_list = []

    # Set up data structures on each GPU.
    for ng in range(NUM_GPUS):
        print("Setting up data structures on GPU", ng)
        cp.cuda.Device(ng).use()
        pk.set_device_id(ng)

        temp_x_ar = cp.random.rand(N)
        temp_v_ar = 0.01*cp.random.rand(N)
        temp_gpu_E_ar = cp.zeros(N)
        temp_gpu_R_ar = cp.zeros(N)

        d_temp_x_ar = pk.array(temp_x_ar)
        d_temp_v_ar = pk.array(temp_v_ar)
        d_temp_gpu_E_ar = pk.array(temp_gpu_E_ar)
        d_temp_gpu_R_ar = pk.array(temp_gpu_R_ar)

        x_ar_list.append(temp_x_ar)
        v_ar_list.append(temp_v_ar)
        gpu_E_ar_list.append(temp_gpu_E_ar)
        gpu_R_ar_list.append(temp_gpu_R_ar)

        d_x_ar_list.append(d_temp_x_ar) 
        d_v_ar_list.append(d_temp_v_ar) 
        d_E_ar_list.append(d_temp_gpu_E_ar)
        d_R_ar_list.append(d_temp_gpu_R_ar)
        
    cpu_E_ar = np.zeros(N)
    
    for ng in range(NUM_GPUS):
        cp.cuda.Device(ng).use()
        print("Beginning average position on GPU", ng, "=", cp.mean(x_ar_list[ng]))
    
    for step in range(num_steps):
        # Spawn a task on each GPU.
        for ng in range(NUM_GPUS):
            cp.cuda.Device(ng).use()
            pk.set_device_id(ng)
    
            # Copy EF data from CPU to GPU.
            gpu_E_ar_list[ng][:] = cp.asarray(cpu_E_ar[:]) 
                    
            # Draw random numbers with CuPy.           
            gpu_R_ar_list[ng][:] = cp.random.rand(N)            
        
            # Advect particles.
            advect(N, d_x_ar_list[ng], d_v_ar_list[ng], d_E_ar_list[ng], d_R_ar_list[ng], threads_per_block, num_blocks)

        cpu_E_ar.fill(0.01 * np.random.rand())

    for ng in range(NUM_GPUS):
        cp.cuda.Device(ng).use()
        print("End average position on GPU", ng, "=", cp.mean(x_ar_list[ng]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--num_gpus", type=int, default=1)
    parser.add_argument("-N", "--num_particles", type=int, default=100000)
    parser.add_argument("-s", "--num_steps", type=int, default=100)
    args = parser.parse_args()
    main(args.num_gpus, args.num_particles, args.num_steps)
