from parla import Parla
from parla.tasks import spawn, TaskSpace, specialize
from parla.devices import cpu, gpu
from typing import Callable, Optional, List, Dict, Tuple
from time import perf_counter
import os
import argparse as ap
import psutil as ps
import numba as nb


def run(function: Callable[[], Optional[TaskSpace]]):
    """
    Parla applications are typically run in a top-level task.
    This function encapsulates the creation of the top-level task and the Parla runtime.

    Args:
        function (Callable[[], Optional[TaskSpace]]): A parla app that (optionally) returns a task space.
    """
    # Start the Parla runtime
    with Parla():
        # Create an encapsulating top-level task to kick off the computation and wait for it to complete.
        @spawn(placement=cpu, vcus=0)
        async def top_level_task():
            # Run the Parla application
            await function()


def save_domain(A: "np.ndarray", filename: str):
    """
    Save the given domain to the given filename.

    Args:
        A (np.ndarray): The matrix to save.
        filename (str): The filename to save to.
    """
    import numpy as np

    np.save(filename, A)


def load_domain(filename: str) -> "np.ndarray":
    """
    Load the domain from the given filename.

    Args:
        filename (str): The filename to load from.

    Returns:
        np.ndarray: The matrix.
    """
    import numpy as np

    return np.load(filename)


def block_domain(
    A: "nd.ndarray", n_blocks: int, boundary_width: int = 1
) -> List["np.ndarray"]:
    """
    Partition the given domain into 1D row-paritioned blocks of the given size.
    Boundaries are extended to include space to copy the boundary data from neighboring blocks.

    Args:
        A (np.ndarray): The domain to partition.
        n_blocks (int): The number of blocks to parition into.
        boundary_width (int): The width of the boundary to include.

    Returns:
        List[np.ndarray]: The partitioned domain.
    """
    import numpy as np

    assert A.shape[0] % n_blocks == 0, "Domain size must be divisible by n_blocks"
    assert boundary_width >= 0, "Boundary width must be non-negative"
    block_size = A.shape[0] // n_blocks
    assert (
        boundary_width < A.shape[0] // n_blocks
    ), "Boundary width must be less than block size"

    height = A.shape[0]
    width = A.shape[1]

    A_blocked = [None for _ in range(n_blocks)]

    for i in range(n_blocks):
        source_start_i = i * block_size
        source_end_i = (i + 1) * block_size

        target_start_i = boundary_width
        target_end_i = boundary_width + block_size

        A_blocked[i] = np.zeros(
            (block_size + 2 * boundary_width, width), dtype=A.dtype, order="F"
        )

        A_blocked[i][target_start_i:target_end_i] = np.copy(
            A[source_start_i:source_end_i], order="F"
        )

    return A_blocked


def unblock_domain(A: List["np.ndarray"], boundary_width: int = 1) -> "np.ndarray":
    """
    Combine the given blocked domain into a single domain.

    Args:
        A (List[np.ndarray]): The blocked domain.
        boundary_width (int): The width of the boundary to exclude.

    Returns:
        np.ndarray: The combined domain.
    """
    import numpy as np

    assert boundary_width >= 0, "Boundary width must be non-negative"
    assert boundary_width < A[0].shape[0] // len(
        A
    ), "Boundary width must be less than block size"

    block_size = A[0].shape[0]
    interior_block_size = A[0].shape[0] - 2 * boundary_width

    height = sum(interior_block_size for block in A)
    width = A[0].shape[1]

    A_combined = np.zeros((height, width), dtype=A[0].dtype, order="F")

    for i in range(len(A)):
        target_start_i = i * interior_block_size
        target_end_i = (i + 1) * interior_block_size

        source_start_i = boundary_width
        source_end_i = interior_block_size + boundary_width

        A_combined[target_start_i:target_end_i] = np.copy(
            A[i][source_start_i:source_end_i], order="F"
        )

    return A_combined


def move_to_gpu(A_blocked: List["np.ndarray"]) -> List["cp.ndarray"]:
    """
    Move the given blocked matrix across the GPUs.
    Distributed blocked row cyclically across all available GPUs.

    Args:
        A_blocked ([List[np.ndarray | PArray]]): The blocked matrix to move.

    Returns:
        List[[cp.ndarray]]: The blocked matrix on the GPUs.
    """
    import cupy as cp
    from parla.array import PArray
    from parla.devices import cpu

    n_gpus = cp.cuda.runtime.getDeviceCount()

    A_gpu = [None for _ in range(len(A_blocked))]
    for i in range(len(A_blocked)):
        with cp.cuda.Device(i % n_gpus) as active_device:
            source = A_blocked[i]
            if isinstance(source, PArray):
                source = source.get(cpu(0))
            A_gpu[i] = cp.asarray(source)
            active_device.synchronize()
    return A_gpu


def move_to_cpu(
    A_blocked: List["cp.ndarray | PArray"],
) -> List["np.ndarray"]:
    """
    Move the given blocked matrix across the CPUs.
    Args:
        A_blocked ([List[cp.ndarray]]): The blocked matrix on the GPU to collect.

    Returns:
        [List[np.ndarray]]: The blocked matrix on the CPU.
    """
    import cupy as cp
    import numpy as np
    from parla.tasks import spawn, AtomicTaskSpace as TaskSpace
    from parla.array import PArray
    from parla.devices import cpu

    A_cpu = [None for _ in range(len(A_blocked))]

    for i in range(len(A_blocked)):
        source = A_blocked[i]
        if isinstance(source, PArray):
            A_cpu[i] = source.get(cpu(0))
        else:
            A_cpu[i] = cp.asnumpy(source)
    return A_cpu


def stream_cupy_to_numba(cp_stream):
    """
    Notes:
        1. The lifetime of the returned Numba stream should be as long as the CuPy one,
           which handles the deallocation of the underlying CUDA stream.
        2. The returned Numba stream is assumed to live in the same CUDA context as the
           CuPy one.
        3. The implementation here closely follows that of cuda.stream() in Numba.
    """
    from ctypes import c_void_p
    import weakref
    from numba import cuda

    # get the pointer to actual CUDA stream
    raw_str = cp_stream.ptr

    # gather necessary ingredients
    ctx = cuda.devices.get_context()
    handle = c_void_p(raw_str)
    finalizer = None  # let CuPy handle its lifetime, not Numba

    # create a Numba stream
    nb_stream = cuda.cudadrv.driver.Stream(weakref.proxy(ctx), handle, finalizer)

    return nb_stream


@nb.njit(parallel=True)
def jacobi(a0, a1):
    """
    CPU code to perform a single step in the Jacobi iteration.
    """
    a1[1:-1, 1:-1] = 0.25 * (
        a0[2:, 1:-1] + a0[:-2, 1:-1] + a0[1:-1, 2:] + a0[1:-1, :-2]
    )

    return a1


def plot_domain(domain: "np.ndarray", filename: str):
    """
    Plot the given domain to the given filename.

    Args:
        domain (np.ndarray): The domain to plot.
        filename (str): The filename to plot to.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.imshow(domain)
    plt.savefig(filename)


if __name__ == "__main__":
    import numpy as np

    domain = np.ones((10, 10), dtype=np.float32)
    boundary_width = 1
    n_blocks = 2
    blocked_domain = block_domain(domain, n_blocks, boundary_width)

    domain_reconstructed = unblock_domain(blocked_domain, boundary_width)
