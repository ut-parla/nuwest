from parla import Parla
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu

from typing import Callable
from time import perf_counter

import numpy as np
import pykokkos as pk

pk.set_default_space(pk.OpenMP)


def run(function: Callable[[], float], print_time: bool = False):
    # Start the Parla runtime
    with Parla():
        # Create an encapsulating top-level task to kick off the computation and wait for it to complete.
        @spawn(placement=cpu, vcus=0)
        async def top_level_task():
            # Run the Parla application and print the time it took if requested.
            start_t = perf_counter()
            await function()
            end_t = perf_counter()

            elapsed = end_t - start_t
            if print_time:
                print(f"Execution time: {elapsed} seconds", flush=True)
            return elapsed


@pk.workunit
def daxpy_kernel(
    tid: int,
    start: int,
    end: int,
    out: pk.View1D[float],
    a: float,
    x: pk.View1D[float],
    y: pk.View1D[float],
    stride: int,
):
    for i in range(start + tid, end, stride):
        out[i] = a * x[i] + y[i]


def daxpy(start, end, iout, a, ix, iy):
    num_threads: int = 1
    pk.parallel_for(
        num_threads,
        daxpy_kernel,
        start=start,
        end=end,
        out=iout,
        a=a,
        x=ix,
        y=iy,
        stride=num_threads,
    )


def compile_daxpy():
    N = 100
    x = np.random.rand(N)
    y = np.random.rand(N)
    out = np.empty_like(x)
    a: float = 2.0
    start: int = 0
    end: int = N

    x_ar = pk.array(x)
    y_ar = pk.array(y)
    out_ar = pk.array(out)

    daxpy(start, end, out_ar, a, x_ar, y_ar)


compile_daxpy()


async def daxpy_example():
    N = 200000000
    x = np.random.rand(N)
    y = np.random.rand(N)
    out = np.empty_like(x)
    truth = np.empty_like(x)

    truth_arr = pk.array(truth)
    x_arr = pk.array(x)
    y_arr = pk.array(y)
    out_arr = pk.array(out)

    a = 2.0
    splits = 2

    start_t = perf_counter()
    # truth[:] = a * x[:] + y[:]
    daxpy(0, N, truth_arr, a, x_arr, y_arr)
    end_t = perf_counter()
    print("Reference: ", end_t - start_t, flush=True)

    start_t = perf_counter()
    T = TaskSpace("Daxpy")
    for i in range(splits):

        @spawn(T[i], placement=cpu, vcus=0)
        def daxpy_task():
            start = i * N // splits
            end = (i + 1) * N // splits
            # out[start:end] = a * x[start:end] + y[start:end]
            daxpy(start, end, out_arr, a, x_arr, y_arr)

    @spawn(T[splits], dependencies=[T[:splits]], placement=cpu, vcus=0)
    def check():
        end_t = perf_counter()
        print("Parla: ", end_t - start_t, flush=True)
        print(out_arr)
        print(truth_arr)
        print("Check: ", np.allclose(out, truth), flush=True)

    await T


run(daxpy_example)
