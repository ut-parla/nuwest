import numpy as np
import sys

import pykokkos as pk

pk.set_default_space(pk.OpenMP)
from advection_kernel import advect


def main():
    threads_per_block = 64
    num_blocks = 1024

    N = 100000
    num_steps = 10

    x_ar = np.random.rand(N)
    v_ar = 0.01 * np.random.rand(N)
    E_ar = np.zeros(N)

    d_x_ar = pk.array(x_ar)
    d_v_ar = pk.array(v_ar)
    d_E_ar = pk.array(E_ar)

    print("\n Launching main Parla task to run for", num_steps, "steps\n")

    advect(N, d_x_ar, d_v_ar, d_E_ar, threads_per_block, num_blocks, 1)


if __name__ == "__main__":
    main()
