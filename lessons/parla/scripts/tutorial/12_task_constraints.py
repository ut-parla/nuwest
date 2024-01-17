from typing import Callable, Optional
from time import perf_counter, sleep

# Handle for Parla runtime
from parla import Parla

# Spawning  tasks
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu

from time import perf_counter, sleep

from helper import run


async def vcu_example():
    T = TaskSpace("T")
    import numpy as np

    N = 10000
    n_tasks = 5

    # Try changing the cost to increase parallelism
    cost = 1  # Serial
    # cost = 1/8 # 2 Active CPU Threads
    # cost = 1/4 # 4 Active CPU Threads

    start_t = perf_counter()
    vectors = [np.random.rand(N) for _ in range(n_tasks)]
    matricies = [np.random.rand(N, N) for _ in range(n_tasks)]

    for i in range(n_tasks):

        @spawn(T[i], placement=cpu, vcus=cost)
        def task():
            print("Starting:", T[i], flush=True)
            v = vectors[i]
            M = matricies[i]
            for _ in range(1):
                v = M @ v
            print("Completed: ", T[i], flush=True)

    @spawn(T["sum"], [T[:n_tasks]], placement=cpu, vcus=cost)
    def sum_task():
        print("Starting sum", flush=True)
        vectors[0] = sum(vectors)
        print("Finished sum", flush=True)

    await T
    end_t = perf_counter()

    print("Elapsed Time: ", end_t - start_t)


run(vcu_example)
