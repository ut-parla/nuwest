from parla import Parla
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu, gpu

from typing import Callable
from time import perf_counter, sleep


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


# A simple task that prints a message
async def my_first_task():
    @spawn()
    def hello():
        print("Hello from the task!")


print("Running my_first_task example...")
run(my_first_task)
