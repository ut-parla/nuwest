from typing import Callable, Optional
from time import perf_counter, sleep

# Handle for Parla runtime
from parla import Parla

# Spawning  tasks
from parla.tasks import spawn, TaskSpace, get_current_task, get_current_context
from parla.devices import cpu, gpu

from time import perf_counter, sleep

from helper import run


async def device_task():
    T = TaskSpace("Device")

    for i in range(2):

        @spawn(T["cpu", i], placement=cpu)
        def cpu_task():
            # Runs on a CPU device
            print(
                f"Hello from {get_current_task()}, running on {get_current_context()}"
            )

        @spawn(T["gpu", i], placement=gpu)
        def gpu_task():
            # Runs on a GPU device
            print(
                f"Hello from {get_current_task()}, running on {get_current_context()}"
            )

        @spawn(T["either", i], placement=[cpu, gpu])
        def either_task():
            # Runs on either a CPU or GPU device
            print(
                f"Hello from {get_current_task()}, running on {get_current_context()}"
            )


run(device_task)
