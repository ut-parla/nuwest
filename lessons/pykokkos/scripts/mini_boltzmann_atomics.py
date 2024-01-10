import argparse
import numpy as np
import cupy as cp

import pykokkos as pk
pk.set_default_space(pk.Cuda)

from advection_kernel_atomics import advect

def main(in_N, in_steps):
    threads_per_block = 64
    num_blocks = 1024

    N = in_N
    num_steps = in_steps

    # Set up data structures
    x_ar = cp.random.rand(N)
    v_ar = 0.01*cp.random.rand(N)
    R_ar = cp.random.rand(N)
    gpu_E_ar = cp.zeros(N)
    lhs_count = cp.zeros(1).astype(int)
    rhs_count = cp.zeros(1).astype(int)
    cpu_E_ar = np.zeros(N)

    print("Beginning average position =", cp.mean(x_ar))
    
    for step in range(num_steps):

        # Draw random numbers with CuPy
        R_ar[:] = cp.random.rand(N)   

        # Generate random electric field
        cpu_E_ar[:] = np.random.rand(N)-0.5
        
        # Copy electric field data from CPU to GPU
        gpu_E_ar[:] = cp.asarray(cpu_E_ar[:]) 
                    
        # PyKokkos kernel for particle advection + collision
        advect(N, x_ar, v_ar, E_ar, R_ar, lhs_count, rhs_count, threads_per_block, num_blocks)

    print("Total number of LHS boundary collisions=",lhs_count[0])
    print("Total number of RHS boundary collisions=",rhs_count[0])
    print("End average position =", cp.mean(x_ar))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num_particles", type=int, default=100000)
    parser.add_argument("-s", "--num_steps", type=int, default=100)
    args = parser.parse_args()
    main(args.num_particles, args.num_steps)
