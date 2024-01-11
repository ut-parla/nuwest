from typing import Callable, Optional
from time import perf_counter, sleep

# Handle for Parla runtime
from parla import Parla

# Spawning  tasks
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu

from time import perf_counter, sleep

from helper import run


async def out_of_order():
    T = TaskSpace("T")

    @spawn(T[1], [T[0]])
    def task1():
        print("Hello from task1!", flush=True)

    @spawn(T[0])
    def task0():
        print("Hello from task0!", flush=True)


run(out_of_order)
