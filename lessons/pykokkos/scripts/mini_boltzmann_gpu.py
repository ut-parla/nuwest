import argparse
import numpy as np
import cupy as cp

import pykokkos as pk
pk.set_default_space(pk.Cuda)

from advection_kernel import advect

def main(in_N, in_steps):
    threads_per_block = 64
    num_blocks = 1024

    N = in_N
    num_steps = in_steps

    # Set up data structures.
    x_ar = cp.random.rand(N)
    v_ar = 0.01*cp.random.rand(N)
    R_ar = cp.random.rand(N)
    gpu_E_ar = cp.zeros(N)
    cpu_E_ar = np.zeros(N)

    print("Beginning average position =", cp.mean(x_ar))
    
    for step in range(num_steps):
        # Draw random numbers with CuPy.
        R_ar[:] = cp.random.rand(N) 
        
        # Generate random electric field.
        cpu_E_ar.fill(0.01 * np.random.rand())
        
        # Copy electric field data from CPU to GPU.
        gpu_E_ar[:] = cp.asarray(cpu_E_ar[:])     
        
        # PyKokkos kernel for particle advection + collision.
        advect(N, x_ar, v_ar, gpu_E_ar, R_ar, threads_per_block, num_blocks)

    print("End average position =", cp.mean(x_ar))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num_particles", type=int, default=100000)
    parser.add_argument("-s", "--num_steps", type=int, default=100)
    args = parser.parse_args()
    main(args.num_particles, args.num_steps)
