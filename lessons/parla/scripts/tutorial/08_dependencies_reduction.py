from typing import Callable, Optional
from time import perf_counter, sleep

# Handle for Parla runtime
from parla import Parla

# Spawning  tasks
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu

from time import perf_counter, sleep

from helper import run


async def task_reduction_dependencies():
    import numpy as np

    T = TaskSpace("T")

    N = 8
    levels = int(np.log2(N))

    array = np.arange(N)
    expected_sum = np.sum(array)

    scratch = {}
    scratch[0] = array
    for level in range(1, levels):
        length = int(N / 2 ** (level + 1))
        scratch[level] = np.zeros(length)

    print("Initial array: ", array, flush=True)
    print("Expected sum: ", expected_sum, flush=True)

    # Generate tasks for a reduction tree
    for level in range(levels):
        stride = int(2 ** (level + 1))
        for idx in range(0, N, stride):
            writes_to = idx
            reads_from = idx + stride // 2

            @spawn(T[level, writes_to], [T[level - 1, reads_from]])
            def task():
                array[writes_to] += array[reads_from]

    # Wait for the reduction tree to finish
    await T[levels - 1, 0]

    print("Final array: ", array, flush=True)
    print("Sum: ", array[0], flush=True)


print("Running task_reduction_dependencies...")
run(task_reduction_dependencies)
