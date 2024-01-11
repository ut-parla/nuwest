from typing import Callable, Optional
from time import perf_counter, sleep

# Handle for Parla runtime
from parla import Parla

# Spawning  tasks
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu

from time import perf_counter, sleep

from helper import run


async def named_tasks():
    T = TaskSpace("T")
    n_tasks = 4
    for i in range(n_tasks):

        @spawn(T[i])
        def task():
            print(f"Hello from {T[i]}! \n", flush=True)

    sleep(0.1)
    print("TasksSpace: ", T)
    print("Contains: ", list(T.tasks), flush=True)


print("Running named_tasks...")
run(named_tasks)
