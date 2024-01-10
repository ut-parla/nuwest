from helper_numpy import set_numpy_threads, parse_solve_args
from typing import List, Dict

args = parse_solve_args()
set_numpy_threads(args.threads)

import numpy as np
from helper import (
    run,
    jacobi as jacobi_step,
    move_to_cpu,
    move_to_gpu,
    load_domain,
    block_domain,
    unblock_domain,
    stream_cupy_to_numba,
    plot_domain,
)
import os
import numba
import time

from parla import Parla
from parla.tasks import spawn, AtomicTaskSpace as TaskSpace, specialize, Tasks
from parla.devices import cpu, gpu
from parla.array import PArray, asarray_batch
from numba import cuda

import cupy as cp


@specialize
@numba.njit(parallel=True)
def jacobi(a0, a1):
    """
    CPU code to perform a single step in the Jacobi iteration.
    """
    a1[1:-1, 1:-1] = 0.25 * (
        a0[2:, 1:-1] + a0[:-2, 1:-1] + a0[1:-1, 2:] + a0[1:-1, :-2]
    )


@jacobi.variant(gpu)
def jacobi_gpu(a0, a1):
    """
    GPU kernel call to perform a single step in the Jacobi iteration.
    """
    threads_per_block_x = 32
    threads_per_block_y = 1024 // threads_per_block_x
    blocks_per_grid_x = (a0.shape[0] + (threads_per_block_x - 1)) // threads_per_block_x
    blocks_per_grid_y = (a0.shape[1] + (threads_per_block_y - 1)) // threads_per_block_y

    cp_stream = cp.cuda.get_current_stream()
    nb_stream = stream_cupy_to_numba(cp_stream)

    gpu_jacobi_kernel[
        (blocks_per_grid_x, blocks_per_grid_y),
        (threads_per_block_x, threads_per_block_y),
        nb_stream,
    ](a0, a1)
    nb_stream.synchronize()


@cuda.jit
def gpu_jacobi_kernel(a0, a1):
    """
    Actual CUDA kernel to do a single step.
    """
    i, j = cuda.grid(2)
    if 0 < i < a1.shape[0] - 1 and 0 < j < a1.shape[1] - 1:
        a1[i, j] = 0.25 * (a0[i - 1, j] + a0[i + 1, j] + a0[i, j - 1] + a0[i, j + 1])


def block_jacobi(
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

    T = TaskSpace("Jacobi")

    for iter in range(iterations):
        for i in range(args.workers):
            # Dependencies
            self = [T[iter - 1, i]]
            left = [T[iter - 1, i - 1]] if i > 0 else []
            right = [T[iter - 1, i + 1]] if i < args.workers - 1 else []
            dependencies = self + left + right

            # Dataflow
            read_interior = [input_domain[i]]
            read_left_boundary = [input_domain[i - 1][-2, :]] if i > 0 else []
            read_right_boundary = (
                [input_domain[i + 1][1, :]] if i < args.workers - 1 else []
            )

            write_interior = [output_domain[i]]

            read = read_interior + read_left_boundary + read_right_boundary
            write = write_interior

            @spawn(
                T[iter, i],
                dependencies=dependencies,
                placement=[gpu],
                input=read,
                inout=write,
            )
            def step():
                print(f"Running iteration {iter} on worker {i}", flush=True)
                interior_read = input_domain[i].array
                interior_write = output_domain[i].array

                if i > 0:
                    input_domain[i][0, :] = input_domain[i - 1][-2, :]
                else:
                    interior_read = input_domain[i][1:, :].array
                    interior_write = output_domain[i][1:, :].array

                if i < args.workers - 1:
                    input_domain[i][-1, :] = input_domain[i + 1][1, :]
                else:
                    interior_read = input_domain[i][:-1, :].array
                    interior_write = output_domain[i][:-1, :].array

                interior_write = jacobi(interior_read, interior_write)

        output_domain, input_domain = input_domain, output_domain

    T.wait()


async def test_blocked_jacobi():
    A = load_domain(args.input)
    n = A.shape[0]
    assert n % args.workers == 0, "Matrix size must be divisible by number of workers"

    block_size = n // args.workers

    A_blocked_in = block_domain(A, n_blocks=args.workers, boundary_width=1)
    A_blocked_out = block_domain(A, n_blocks=args.workers, boundary_width=1)

    A_blocked_in = asarray_batch(A_blocked_in)
    A_blocked_out = asarray_batch(A_blocked_out)

    start_t = time.perf_counter()
    block_jacobi(A_blocked_in, A_blocked_out, iterations=args.max_iterations)
    end_t = time.perf_counter()

    print(f"Time: {end_t - start_t:.4f}", flush=True)
    output_blocked = move_to_cpu(A_blocked_out)
    output = unblock_domain(output_blocked, boundary_width=1)

    plot_domain(output, "domain.png")


if __name__ == "__main__":
    run(test_blocked_jacobi)
