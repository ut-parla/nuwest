import argparse
import numpy as np
import cupy as cp

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

    # Set up data structures on each GPU.
    x_ar_list = []
    v_ar_list = []
    gpu_E_ar_list = []
    R_ar_list = []
    
    for ng in range(NUM_GPUS):
        print("Setting up data structures on GPU", ng)
        cp.cuda.Device(ng).use()
        pk.set_device_id(ng)

        temp_x_ar = cp.random.rand(N)
        temp_v_ar = 0.01*cp.random.rand(N)
        temp_gpu_E_ar = cp.zeros(N)
        temp_R_ar = cp.zeros(N)

        x_ar_list.append(temp_x_ar)
        v_ar_list.append(temp_v_ar)
        gpu_E_ar_list.append(temp_gpu_E_ar)
        R_ar_list.append(temp_R_ar)

    cpu_E_ar = np.zeros(N)
    
    for ng in range(NUM_GPUS):
        cp.cuda.Device(ng).use()
        print("Beginning average position on GPU", ng, "=", cp.mean(x_ar_list[ng]))
    
    for step in range(num_steps):
        # Generate random electric field.
        cpu_E_ar.fill(0.01 * np.random.rand())
        
        # Spawn a task on each GPU.
        for ng in range(NUM_GPUS):
            
            cp.cuda.Device(ng).use()
            pk.set_device_id(ng)

            # Draw random numbers with CuPy.
            R_ar_list[ng][:] = cp.random.rand(N)  
            
            # Copy electric field data from CPU to GPU.
            gpu_E_ar_list[ng][:] = cp.asarray(cpu_E_ar[:]) 
                     
            # PyKokkos kernel for particle advection + collision.
            advect(N, x_ar_list[ng], v_ar_list[ng], gpu_E_ar_list[ng], R_ar_list[ng], threads_per_block, num_blocks)

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
