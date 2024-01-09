import argparse
import numpy as np
import cupy as cp
from numba import cuda
import sys
import pykokkos as pk
pk.set_default_space(pk.Cuda)

from advection_kernel import advect


def main(in_N, in_s):
    threads_per_block = 64
    num_blocks = 1024

    N = in_N
    num_steps = in_s

    # Set up data structures on each GPU.
    x_ar = cp.random.rand(N)
    v_ar = 0.01*cp.random.rand(N)
    R_ar = cp.random.rand(N)
    gpu_E_ar = cp.zeros(N)
    lhs_count = cp.zeros(1).astype(int)
    rhs_count = cp.zeros(1).astype(int)

    d_x_ar = pk.array(x_ar)
    d_v_ar = pk.array(v_ar)
    d_R_ar = pk.array(R_ar)
    d_E_ar = pk.array(gpu_E_ar)
    d_lhs_count = pk.array(lhs_count)
    d_rhs_count = pk.array(rhs_count)

    cpu_E_ar = np.zeros(N)

    print("Beginning average position =", cp.mean(x_ar))
    
    for step in range(num_steps):
        # Copy EF data from CPU to GPU.
        gpu_E_ar[:] = cp.asarray(cpu_E_ar[:]) 
                    
        # Draw random numbers with CuPy.
        R_ar[:] = cp.random.rand(N)            
        
        # Advect particles.
        advect(N, d_x_ar, d_v_ar, d_E_ar, d_R_ar, d_lhs_count, d_rhs_count, threads_per_block, num_blocks)

        cpu_E_ar[:] = np.random.rand(N)-0.5

    print("Total number of LHS boundary collisions=",lhs_count[0])
    print("Total number of RHS boundary collisions=",rhs_count[0])
    print("End average position =", cp.mean(x_ar))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num_particles", type=int, default=100000)
    parser.add_argument("-s", "--num_steps", type=int, default=100)
    args = parser.parse_args()
    main(args.num_particles, args.num_steps)
