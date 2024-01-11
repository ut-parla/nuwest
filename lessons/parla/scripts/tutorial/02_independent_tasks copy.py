from parla import Parla
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu

from helper import run

from typing import Callable
from time import perf_counter

import numpy as np


async def independent_tasks():
    n_tasks = 4
    for i in range(n_tasks):

        @spawn()
        def task():
            # Local variables are captured by a shallow copy of everythig in the closure of `task()`
            # i is captured by value and is not shared between tasks
            print(f"Hello from Task {i}! \n", flush=True)
            # flush=True is needed to ensure that the print statement is not buffered.


print("Running independent_tasks example...")
run(independent_tasks)
