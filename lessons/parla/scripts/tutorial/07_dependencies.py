from typing import Callable, Optional
from time import perf_counter, sleep

# Handle for Parla runtime
from parla import Parla

# Spawning  tasks
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu

from time import perf_counter, sleep

from helper import run


async def serial_tasks():
    T = TaskSpace("T")

    for i in range(4):

        @spawn(
            T[i], dependencies=[T[i - 1]]
        )  # Could also have written dependencies=T[:i]
        def task():
            print(f"Hello from {T[i]}! \n", flush=True)


print("Running serial_tasks...")
run(serial_tasks)
