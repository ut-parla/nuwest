from parla import Parla
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu

from helper import run

from typing import Callable
from time import perf_counter

import numpy as np
from time import sleep


async def independent_tasks_dictionary():
    n_tasks = 4
    shared_dict = {}
    for i in range(n_tasks):

        @spawn()
        def task():
            # Local variables are captured by a shallow copy of everything in the closure of `task()`
            # Python primitives are thread safe (locking)
            shared_dict[i] = i**2

    # For now, we need to sleep to ensure that the tasks have completed
    # Later, we'll discuss barriers, returns, and general control flow
    sleep(0.1)
    print("Shared Dictionary: ", shared_dict)


print("Running independent_tasks_dictionary example...")
run(independent_tasks_dictionary)
