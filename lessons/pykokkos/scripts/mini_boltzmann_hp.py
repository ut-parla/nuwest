import argparse
import numpy as np
import pykokkos as pk

from advection_kernel import advect

def main(N, num_steps):
    pk.set_default_space(pk.OpenMP)

    threads_per_block = 64
    num_blocks = 1024

    # Set up data structures.
    x_ar = np.random.rand(N)
    v_ar = 0.01*np.random.rand(N)
    E_ar = np.zeros(N)
    R_ar = np.zeros(N)
    
    print("Beginning average position =",np.mean(x_ar))

    for step in range(num_steps):

        # Draw random numbers.
        R_ar[:] = np.random.rand(N)

        # Generate random electric field.
        E_ar.fill(0.01 * np.random.rand())
        
        # PyKokkos kernel for particle advection + collision.
        advect(N, x_ar, v_ar, E_ar, R_ar, threads_per_block, num_blocks)

    print("End average position =", np.mean(x_ar))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num_particles", type=int, default=100000)
    parser.add_argument("-s", "--num_steps", type=int, default=100)
    args = parser.parse_args()
    main(args.num_particles, args.num_steps)
