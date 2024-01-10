import time
from typing import Optional, List
from functools import partial
import os
from helper_numpy import (
    set_numpy_threads,
    parse_factorize_args,
)


args = parse_factorize_args()
set_numpy_threads(args.threads)
import numpy as np
import scipy as sp

try:
    import cupy as cp

    cublas = cp.cuda.cublas
except ImportError:
    raise ImportError("Could not import cupy. NOTE: This examples requires a GPU.")

from helper import (
    load_matrix,
    block_matrix,
    unblock_matrix,
    move_to_cpu,
    move_to_gpu,
    save_matrix,
    create_and_save_truth,
    run,
)

from parla import Parla
from parla.tasks import TaskSpace, specialize, spawn as spawn_task
from parla.devices import cpu, gpu
from parla.array import clone_here, copy, asarray_batch


# Define the linear algebra kernels
# ------------------------------------

# CPU Kernels


# A -> L @ L^T
@specialize
def potrf(A: np.ndarray) -> np.ndarray:
    A = sp.linalg.cholesky(A, lower=True)
    return A


# B -> A^-T B
@specialize
def trsm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.array(A, order="F", dtype=np.float64)
    B = np.array(B.T, order="F", dtype=np.float64)
    B = sp.linalg.blas.dtrsm(1.0, A, B, trans_a=0, lower=1, side=0)
    return B.T


# C -> C - A @ B.T
@specialize
def gemm(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    C -= A @ B.T
    return C


syrk = gemm

# GPU Kernels


@potrf.variant(gpu)
def potrf_gpu(A: cp.ndarray) -> cp.ndarray:
    A = cp.linalg.cholesky(A)
    return A


@trsm.variant(gpu)
def trsm_gpu(A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
    cublas_handle = cp.cuda.device.get_cublas_handle()
    uplo = cublas.CUBLAS_FILL_MODE_LOWER
    A = cp.asarray(A, dtype=np.float64, order="F")
    B = cp.asarray(B, dtype=np.float64, order="F")

    trans = cublas.CUBLAS_OP_T
    side = cublas.CUBLAS_SIDE_RIGHT
    diag = cublas.CUBLAS_DIAG_NON_UNIT

    m, n = (B.side, 1) if B.ndim == 1 else B.shape
    alpha = np.array(1, dtype=A.dtype)
    # Cupy >= 9 requires pointers even for coefficients.
    cublas.dtrsm(
        cublas_handle,
        side,
        uplo,
        trans,
        diag,
        m,
        n,
        alpha.ctypes.data,
        A.data.ptr,
        m,
        B.data.ptr,
        m,
    )
    return B


@gemm.variant(gpu)
def gemm_gpu(A: cp.ndarray, B: cp.ndarray, C: cp.ndarray) -> cp.ndarray:
    C -= A @ B.T
    return C


# Define Cholesky Application
def block_cholesky(A: List[List[cp.ndarray]]) -> TaskSpace:
    """
    Factorize the given matrix.
    Args:
        A (List[List[np.ndarray]]): A is a blocked SPD matrix
    """
    n_blocks = len(A)

    # Define task spaces
    SYRK = TaskSpace("SYRK")
    POTRF = TaskSpace("POTRF")
    TRSM = TaskSpace("TRSM")
    GEMM = TaskSpace("GEMM")

    vcus = 1
    spawn = partial(spawn_task, placement=[gpu, cpu], vcus=vcus)

    for j in range(n_blocks):
        for k in range(j):

            @spawn(
                SYRK[j, k],
                dependencies=[TRSM[j, k], SYRK[j, 0:k]],
                input=[A[j][k]],
                inout=[A[j][j]],
            )
            def t1():
                A[j][j].array[:] = syrk(A[j][k].array, A[j][k].array, A[j][j].array)

        @spawn(
            POTRF[j],
            dependencies=[SYRK[j, 0:j]],
            inout=[A[j][j]],
        )
        def t2():
            A[j][j].array[:] = potrf(A[j][j].array)

        for i in range(j + 1, n_blocks):
            for k in range(j):

                @spawn(
                    GEMM[i, j, k],
                    dependencies=[TRSM[j, k], TRSM[i, k], GEMM[i, j, 0:k]],
                    input=[A[i][k], A[j][k]],
                    inout=[A[i][j]],
                )
                def t3():
                    A[i][j].array[:] = gemm(A[i][k].array, A[j][k].array, A[i][j].array)

            @spawn(
                TRSM[i, j],
                dependencies=[GEMM[i, j, 0:j], POTRF[j]],
                input=[A[j][j]],
                inout=[A[i][j]],
            )
            def t4():
                A[i][j].array[:] = trsm(A[j][j].array, A[i][j].array)

    return POTRF[n_blocks - 1]


async def test_blocked_cholesky():
    A = load_matrix(args.input)
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], "Matrix must be square"
    assert n % args.blocks == 0, "Matrix size must be divisible by block size"

    block_size = n // args.blocks
    A_blocked = block_matrix(A, block_size)
    A_blocked_gpu = move_to_gpu(A_blocked)
    A_blocked_gpu = asarray_batch(A_blocked_gpu)

    start_t = time.perf_counter()
    await block_cholesky(A_blocked_gpu)
    end_t = time.perf_counter()

    L_computed = move_to_cpu(A_blocked_gpu)
    del A_blocked_gpu

    print(f"Time: {end_t - start_t:.4f}", flush=True)

    if args.verify:
        L = unblock_matrix(L_computed, A)
        del L_computed
        L = np.tril(L)

        truth_file = os.path.splitext(args.input)[0] + "_ref.npy"
        L_truth = load_matrix(truth_file)

        try:
            print("Verifying...", flush=True)
            np.testing.assert_allclose(L, L_truth, rtol=1e-4, atol=1e-4)
            print("Verification passed!", flush=True)
        except AssertionError:
            print("Verification failed!", flush=True)


if __name__ == "__main__":
    run(test_blocked_cholesky)
