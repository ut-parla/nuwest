"""
This is a reference implementation of CPU block Cholesky factorization.
"""

import argparse as ap
import os
import time
from typing import List, Dict, Tuple
from helper_numpy import (
    set_numpy_threads,
    parse_factorize_args,
)

args = parse_factorize_args()
set_numpy_threads(args.threads)
import numpy as np
import scipy as sp

# Define the linear algebra kernels
# ------------------------------------


# A -> L @ L^T
def potrf(A: np.ndarray) -> np.ndarray:
    A = sp.linalg.cholesky(A, lower=True)
    return A


# B -> A^-T B
def trsm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.array(A, order="F", dtype=np.float64)
    B = np.array(B.T, order="F", dtype=np.float64)
    B = sp.linalg.blas.dtrsm(1.0, A, B, trans_a=0, lower=1, side=0)
    return B.T


# C -> C - A @ B.T
def gemm(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    C -= A @ B.T
    return C


syrk = gemm


# Define Cholesky Application
def block_cholesky(A: List[List[np.ndarray]]):
    """
    Factorize the given matrix.
    Args:
        A (List[List[np.ndarray]]): A is a blocked SPD matrix
    """
    n_blocks = len(A)

    for j in range(n_blocks):
        for k in range(j):
            A[j][j] = syrk(A[j][k], A[j][k], A[j][j])

        A[j][j] = potrf(A[j][j])

        for i in range(j + 1, n_blocks):
            for k in range(j):
                A[i][j] = gemm(A[i][k], A[j][k], A[i][j])

            A[i][j] = trsm(A[j][j], A[i][j])

    return A


if __name__ == "__main__":
    from helper import (
        load_matrix,
        block_matrix,
        unblock_matrix,
        save_matrix,
        create_and_save_truth,
    )

    A = load_matrix(args.input)
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], "Matrix must be square"
    assert n % args.blocks == 0, "Matrix size must be divisible by block size"

    block_size = n // args.blocks
    A_blocked = block_matrix(A, block_size)

    start_t = time.perf_counter()
    L_computed = block_cholesky(A_blocked)
    end_t = time.perf_counter()
    print(f"Time: {end_t - start_t:.4f} seconds", flush=True)

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
