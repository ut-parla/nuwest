from typing import Callable, Optional
from time import perf_counter, sleep

# Handle for Parla runtime
from parla import Parla

# Spawning  tasks
from parla.tasks import (
    spawn,
    TaskSpace,
    specialize,
    get_current_task,
    get_current_context,
)
from parla.devices import cpu, gpu

from time import perf_counter, sleep

from helper import run

from parla.array import clone_here
import numpy as np
import cupy as cp

from parla.array import asarray as parla_asarray


async def parray_example():
    T = TaskSpace("T")

    A = np.ones(5)
    A = parla_asarray(A)

    @spawn(T[0], placement=[cpu if np.random.rand() < 0.5 else gpu], input=[A])
    def task():
        print(get_current_task(), " running on ", get_current_context())
        print(
            "A is a",
            "Numpy Array"
            if isinstance(A.array, np.ndarray)
            else f"Cupy Array on GPU[{A.array.device}]",
            flush=True,
        )
        A.print_overview()
        print(A.array)
        print("\n")

    @spawn(T[1], [T[0]], placement=[cpu if np.random.rand() < 0.5 else gpu], inout=[A])
    def task():
        print(get_current_task(), " running on ", get_current_context())
        print(
            "A is a",
            "Numpy Array"
            if isinstance(A.array, np.ndarray)
            else f"Cupy Array on GPU[{A.array.device}]",
            flush=True,
        )
        A.print_overview()
        A[:] = A + 1
        print("\n")

    @spawn(
        T[2],
        [T[0], T[1]],
        placement=[cpu if np.random.rand() < 0.5 else gpu],
        inout=[A],
    )
    def task():
        print(get_current_task(), " running on ", get_current_context())
        print(
            "A is a",
            "Numpy Array"
            if isinstance(A.array, np.ndarray)
            else f"Cupy Array on GPU[{A.array.device}]",
            flush=True,
        )
        A.print_overview()
        print(A.array)
        print("\n")


run(parray_example)
