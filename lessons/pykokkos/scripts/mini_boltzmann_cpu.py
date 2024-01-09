import argparse
import numpy as np

import pykokkos as pk
pk.set_default_space(pk.OpenMP)

from advection_kernel import advect

def main(in_N, in_steps):
    threads_per_block = 64
    num_blocks = 1024

    N = in_N
    num_steps = in_steps

    # Set up data structures on CPU
    x_ar = np.random.rand(N)
    v_ar = 0.01*np.random.rand(N)
    E_ar = np.zeros(N)
    R_ar = np.random.rand(N)

    # Set up PyKokkos wrappers
    d_x_ar = pk.array(x_ar)
    d_v_ar = pk.array(v_ar)
    d_E_ar = pk.array(E_ar)
    d_R_ar = pk.array(R_ar)

    print("Beginning average position =",np.mean(x_ar))

    for step in range(num_steps):

        # Draw random numbers with CuPy
        R_ar[:] = np.random.rand(N)

        # Generate random electric field
        E_ar.fill(0.01 * np.random.rand())
        
        # PyKokkos kernel for particle advection + collision
        advect(N, d_x_ar, d_v_ar, d_E_ar, d_R_ar, threads_per_block, num_blocks)

    print("End average position =", np.mean(x_ar))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num_particles", type=int, default=100000)
    parser.add_argument("-s", "--num_steps", type=int, default=100)
    args = parser.parse_args()
    main(args.num_particles, args.num_steps)
