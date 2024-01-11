from parla import Parla
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu

from helper import run

from typing import Callable
from time import perf_counter

import numpy as np
from parla.tasks import get_current_task
from time import sleep


async def anonymous_tasks():
    list_of_tasks = []
    for i in range(4):

        @spawn()
        def task():
            my_name = get_current_task()
            print(f"Hello from Task {my_name}! \n", flush=True)

        list_of_tasks.append(task)

    sleep(0.1)

    print("List of Tasks: ", list_of_tasks, flush=True)


print("Running anonymous_tasks...")
run(anonymous_tasks)
