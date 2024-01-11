from typing import Callable, Optional
from time import perf_counter, sleep

# Handle for Parla runtime
from parla import Parla

# Spawning  tasks
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu

from time import perf_counter, sleep

from helper import run


async def simple_barrier():
    T = TaskSpace("T")

    for i in range(4):

        @spawn(T[i])
        def task1():
            print(f"Hello from task {i}!", flush=True)

        await T


run(simple_barrier)
