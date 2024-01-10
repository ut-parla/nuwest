from helper_numpy import set_numpy_threads, parse_build_domain_args

args = parse_build_domain_args()
set_numpy_threads(args.threads)

import numpy as np
from helper import jacobi as jacobi_step, save_domain
import os


def build_domain(width: int, height: int, val: int = 1000) -> np.ndarray:
    """
    Build a domain of the given size.

    Args:
        width (int): The width of the domain.
        height (int): The height of the domain.

    Returns:
        np.ndarray: The domain.
    """
    domain = np.zeros((height, width), dtype=np.float32)
    domain[:, 0] = val
    domain[:, -1] = val
    return domain


def solve_jacobi(domain: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """
    Solve the given domain using the Jacobi method.

    Args:
        domain (np.ndarray): The domain to solve.
        max_iterations (int): The maximum number of iterations to perform.

    Returns:
        np.ndarray: The solved domain.
    """

    temp_domain = np.copy(domain)

    for _ in range(max_iterations):
        temp_domain = jacobi_step(domain, temp_domain)
        domain, temp_domain = temp_domain, domain
    return domain


if __name__ == "__main__":
    domain = build_domain(args.width, args.height)
    save_domain(domain, args.output)

    truth_file = os.path.splitext(args.output)[0] + "_ref.npy"
    solution = solve_jacobi(domain, args.max_iterations)
    save_domain(solution, truth_file)
