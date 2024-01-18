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


def clone_here_copy_example_wrapper():
    import numpy as np
    import cupy as cp

    M = 5
    N = 5
    A = np.random.rand(M)
    B = cp.arange(N)

    def clone_here_copy_example():
        T = TaskSpace("T")

        @spawn(placement=[cpu if np.random.rand() < 0.5 else gpu])
        def task():
            print(get_current_task(), " running on ", get_current_context())
            C = clone_here(A)
            print(
                "C is a",
                "Numpy Array"
                if isinstance(C, np.ndarray)
                else f"Cupy Array on GPU[{C.device}]",
                flush=True,
            )

        return T

    run(clone_here_copy_example)


clone_here_copy_example_wrapper()
