from helper_numpy import set_numpy_threads, parse_solve_args
from typing import List, Dict

args = parse_solve_args()
set_numpy_threads(args.threads)

import numpy as np
from helper import (
    run,
    # jacobi as jacobi_step,
    move_to_cpu,
    move_to_gpu,
    load_domain,
    block_domain,
    unblock_domain,
    stream_cupy_to_numba,
    plot_domain,
)
import os
import time

from parla import Parla
from parla.tasks import spawn, AtomicTaskSpace as TaskSpace, specialize, Tasks
from parla.devices import cpu, gpu
from parla.array import PArray, asarray_batch

import cupy as cp


async def block_jacobi(
    input_domain: List[PArray],
    output_domain: List[PArray],
    iterations: int = args.max_iterations,
):
    """
    Run the Jacobi iteration on the given domain.

    Args:
        input_domain (np.ndarray): The domain to start with.
        output_domain (np.ndarray): The domain to write the result to.
        iterations (int): The number of iterations to run.
    """

    assert iterations % 2 == 0, "Number of iterations must be even"

    for i in range(args.workers):
        input_domain[i].set_name(f"input_domain_{i}")
        output_domain[i].set_name(f"output_domain_{i}")
    shape = input_domain[i].shape
    print(f"Shape: {shape}", flush=True)
    size = shape[0]

    T = TaskSpace("Jacobi")
    for iter in range(iterations):
        for i in range(args.workers):
            # Dependencies
            if iter > 0:
                self = [T[:iter, i]]
                left = [T[:iter, i - 1]] if i > 0 else []
                right = [T[:iter, i + 1]] if i < args.workers - 1 else []
                dependencies = self + left + right
            else:
                dependencies = []

            # Dataflow
            read_interior = [input_domain[i]]
            read_left_boundary = [input_domain[i - 1][size - 2]] if i > 0 else []
            read_right_boundary = (
                [input_domain[i + 1][1]] if i < args.workers - 1 else []
            )

            write_interior = [output_domain[i]]

            read = read_interior + read_left_boundary + read_right_boundary
            write = write_interior

            @spawn(
                T[iter, i],
                # T[iter, i],
                # dependencies=dependencies + [T[iter, i - 1]],
                placement=[cpu if iter % 2 == 0 else gpu],
                input=read,
                inout=write,
            )
            def step():
                print(f"Running iteration {iter} on worker {i}", flush=True)
                interior_read = input_domain[i].array
                interior_write = output_domain[i].array

                dummy = np.zeros_like(interior_read)

                if i > 0:
                    dummy[0] = input_domain[i - 1][size - 2].array
                else:
                    interior_read = input_domain[i][1:].array
                    interior_write = output_domain[i][1:].array

                if i < args.workers - 1:
                    dummy[size - 1] = input_domain[i + 1][1].array
                else:
                    interior_read = input_domain[i][: (size - 1)].array
                    interior_write = output_domain[i][: (size - 1)].array

                # interior_write = jacobi(interior_read, interior_write)

        await T

        output_domain, input_domain = input_domain, output_domain

        # T.wait()


async def test_blocked_jacobi():
    A = load_domain(args.input)
    n = A.shape[0]
    assert n % args.workers == 0, "Matrix size must be divisible by number of workers"

    block_size = n // args.workers

    A_blocked_in = block_domain(A, n_blocks=args.workers, boundary_width=1)
    A_blocked_out = block_domain(A, n_blocks=args.workers, boundary_width=1)

    A_blocked_in = asarray_batch(A_blocked_in, base="in")
    A_blocked_out = asarray_batch(A_blocked_out, base="out")

    start_t = time.perf_counter()
    await block_jacobi(A_blocked_in, A_blocked_out, iterations=args.max_iterations)
    end_t = time.perf_counter()

    print(f"Time: {end_t - start_t:.4f}", flush=True)
    output_blocked = move_to_cpu(A_blocked_out)
    output = unblock_domain(output_blocked, boundary_width=1)

    plot_domain(output, "domain.png")


if __name__ == "__main__":
    run(test_blocked_jacobi)
