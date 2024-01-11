from typing import Callable, Optional
from time import perf_counter, sleep

# Handle for Parla runtime
from parla import Parla

# Spawning  tasks
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu

from time import perf_counter, sleep

from helper import run


async def taskspace_slicing():
    T = TaskSpace("T")

    for i in range(2):
        for j in range(2):

            @spawn(T[i, j])
            def task():
                print(f"Hello from {T[i, j]}! \n", flush=True)

    sleep(0.1)
    print("TasksSpace: ", T)
    print("Slice of Tasks: ", T[0:1, 0:2], flush=True)
    print("State of Task[0, 0]: ", T[0, 0].state, flush=True)


print("Running taskspace_slicing...")
run(taskspace_slicing)
