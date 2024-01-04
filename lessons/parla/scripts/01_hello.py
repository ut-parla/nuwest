from parla import Parla
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu, gpu

from typing import Callable
from time import perf_counter


def run(function: Callable[[], float], print_time: bool = False):
    # Start the Parla runtime
    with Parla():
        # Create an encapsulating top-level task to kick off the computation and wait for it to complete.
        @spawn(placement=cpu, vcus=0)
        async def top_level_task():
            # Run the Parla application and print the time it took if requested.
            start_t = perf_counter()
            await function()
            end_t = perf_counter()

            elapsed = end_t - start_t
            if print_time:
                print(f"Execution time: {elapsed} seconds", flush=True)
            return elapsed


async def first_example():
    @spawn()
    def task_hello():
        print("Hello!", flush=True)

    @spawn()
    def task_goodbye():
        print("Goodbye!", flush=True)


run(first_example)
