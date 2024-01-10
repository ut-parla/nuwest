import argparse as ap
import os
from typing import List, Dict, Tuple, Optional
from helper_numpy import (
    set_numpy_threads,
    parse_build_matrix_args,
)
import psutil as ps


def build_matrix(size: int, seed: int = 0) -> "np.ndarray":
    """
    Build a symmetric positive definite matrix of the given size.

    Args:
        size (int): The size of the matrix to build.

    Returns:
        np.ndarray: The matrix.
    """
    import numpy as np

    np.random.seed(seed)
    A = np.random.randn(size, size)
    C = A @ A.T
    return C


if __name__ == "__main__":
    args = parse_build_matrix_args()
    set_numpy_threads(args.threads)

    from helper import (
        save_matrix,
        create_and_save_truth,
    )

    A = build_matrix(args.n, args.seed)
    save_matrix(A, args.output)

    truth_file = os.path.splitext(args.output)[0] + "_ref.npy"
    L = create_and_save_truth(A, truth_file)
