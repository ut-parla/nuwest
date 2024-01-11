from parla import Parla
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu, gpu
from typing import Callable, Optional, List, Dict, Tuple
from time import perf_counter
import os
import argparse as ap
import psutil as ps


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


def save_matrix(A: "np.ndarray", filename: str):
    """
    Save the given matrix to the given filename.

    Args:
        A (np.ndarray): The matrix to save.
        filename (str): The filename to save to.
    """
    import numpy as np

    np.save(filename, A)


def load_matrix(filename: str) -> "np.ndarray":
    """
    Load the matrix from the given filename.

    Args:
        filename (str): The filename to load from.

    Returns:
        np.ndarray: The matrix.
    """
    import numpy as np

    return np.load(filename)


def block_matrix(A: "np.ndarray", block_size: int) -> List[List["np.ndarray"]]:
    """
    Partition the given matrix into blocks of the given size.

    Args:
        A (np.ndarray): The matrix to partition.
        block_size (int): The size of the blocks to partition into.

    Returns:
        List[List[np.ndarray]]: The partitioned matrix.
    """
    import numpy as np

    n_blocks = A.shape[0] // block_size
    blocks = [[None for _ in range(n_blocks)] for _ in range(n_blocks)]

    for i in range(n_blocks):
        start_i = i * block_size
        end_i = (i + 1) * block_size

        for j in range(n_blocks):
            start_j = j * block_size
            end_j = (j + 1) * block_size
            blocks[i][j] = np.copy(A[start_i:end_i, start_j:end_j], order="F")

    return blocks


def unblock_matrix(
    blocks: List[List["np.ndarray | PArray"]], out: Optional["np.ndarray"] = None
) -> "np.ndarray":
    """
    Reconstruct a matrix from its blocks.

    Args:
        blocks (List[List[np.ndarray | PArray]]): The blocks of the matrix.

    Returns:
        np.ndarray: The reconstructed matrix.
    """
    import numpy as np
    from parla.array import PArray
    from parla.devices import cpu

    if isinstance(blocks[0][0], PArray):
        block_size = blocks[0][0].get(cpu(0)).shape[0]
    else:
        block_size = blocks[0][0].shape[0]
    n = len(blocks) * block_size

    if out is None:
        A = np.empty((n, n))
    else:
        A = out

    for i in range(len(blocks)):
        start_idx_i = i * block_size
        end_idx_i = (i + 1) * block_size

        for j in range(len(blocks)):
            start_idx_j = j * block_size
            end_idx_j = (j + 1) * block_size

            source = blocks[i][j]
            if isinstance(source, PArray):
                source = source.get(cpu(0))

            A[start_idx_i:end_idx_i, start_idx_j:end_idx_j] = source

    return A


def move_to_gpu(A_blocked: List[List["np.ndarray"]]) -> List[List["cp.ndarray"]]:
    """
    Move the given blocked matrix across the GPUs.
    Distributed blocked row cyclically across all available GPUs.

    Args:
        A_blocked (List[List[np.ndarray | PArray]]): The blocked matrix to move.

    Returns:
        List[List[cp.ndarray]]: The blocked matrix on the GPUs.
    """
    import cupy as cp
    from parla.array import PArray
    from parla.devices import cpu

    n_gpus = cp.cuda.runtime.getDeviceCount()

    A_gpu = [[None for _ in range(len(A_blocked))] for _ in range(len(A_blocked))]
    for i in range(len(A_blocked)):
        for j in range(len(A_blocked)):
            with cp.cuda.Device(i % n_gpus) as active_device:
                source = A_blocked[i][j]
                if isinstance(source, PArray):
                    source = source.get(cpu(0))
                A_gpu[i][j] = cp.asarray(source)
                active_device.synchronize()
    return A_gpu


def move_to_cpu(
    A_blocked: List[List["cp.ndarray | PArray"]],
) -> List[List["np.ndarray"]]:
    """
    Move the given blocked matrix across the CPUs.
    Args:
        A_blocked (List[List[cp.ndarray]]): The blocked matrix on the GPU to collect.

    Returns:
        List[List[np.ndarray]]: The blocked matrix on the CPU.
    """
    import cupy as cp
    import numpy as np
    from parla.tasks import spawn, AtomicTaskSpace as TaskSpace
    from parla.array import PArray
    from parla.devices import cpu

    A_cpu = [[None for _ in range(len(A_blocked))] for _ in range(len(A_blocked))]

    for i in range(len(A_blocked)):
        for j in range(len(A_blocked)):
            source = A_blocked[i][j]
            if isinstance(source, PArray):
                A_cpu[i][j] = source.get(cpu(0))
            else:
                A_cpu[i][j] = cp.asnumpy(source)
    return A_cpu


def reference_cholesky(A: "np.ndarray") -> "np.ndarray":
    import numpy as np
    import scipy.linalg as la

    return la.cholesky(A, lower=True)


def create_and_save_truth(A: "np.ndarray", outfile: str) -> "np.ndarray":
    L = reference_cholesky(A)
    save_matrix(L, outfile)
    return L
