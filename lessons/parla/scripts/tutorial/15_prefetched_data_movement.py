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
    A = np.random.rand(5)
    A = parla_asarray(A)

    @spawn(placement=[cpu if np.random.rand() < 0.5 else gpu], input=[A])
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
        # There is a valid copy of A on both devices


run(parray_example)
