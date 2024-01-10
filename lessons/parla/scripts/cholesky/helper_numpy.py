from typing import Callable, Optional, List, Dict, Tuple
from time import perf_counter
import os
import argparse as ap
import psutil as ps


def set_numpy_threads(threads: int = 1):
    """
    Numpy can be configured to use multiple threads for linear algebra operations.
    The backend used by numpy can vary by installation.
    This function attempts to set the number of threads for the most common backends.
    MUST BE CALLED BEFORE IMPORTING NUMPY.

    Args:
        threads (int, optional): The number of threads to use. Defaults to 1.
    """

    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)

    try:
        # Controlling the MKL backend can use mkl and mkl-service modules if installed.
        # preferred method for controlling the MKL backend.
        import mkl

        mkl.set_num_threads(threads)
    except ImportError:
        pass


def parse_build_matrix_args() -> ap.Namespace:
    parser = ap.ArgumentParser()
    parser.add_argument(
        "-n",
        "--size",
        type=int,
        default=1000,
        help="The size of the matrix to build",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed for the random number generator",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="The number of threads to use for numpy",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="matrix.npy",
        help="The output file for the matrix",
    )
    args = parser.parse_args()
    return args


def parse_factorize_args() -> ap.Namespace:
    parser = ap.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers (CPUs / GPUs / Total Streams)",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=2,
        help="Number of blocks (along each dim) to partition the matrix into",
    )
    parser.add_argument(
        "--threads-per-task",
        dest="threads",
        type=int,
        default=1,
        help="Number of threads per task (for CPU tasks)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="matrix.npy",
        help="The input file for the matrix",
    )
    parser.add_argument(
        "--verify",
        "-v",
        action="store_true",
        help="Verify the result against the reference implementation",
    )
    args = parser.parse_args()
    return args
