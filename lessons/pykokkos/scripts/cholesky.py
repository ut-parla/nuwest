import cupy as cp
import numpy as np
import pykokkos as pk

# Code taken from https://github.com/spcl/npbench and then revised to
# show cp vs pk usage.

def cho_pk(A):
    A[0][0] = pk.sqrt(A[0][0])
    for i in range(1, A.shape[0]):
        for j in range(i):
            A[i][j] -= pk.dot(A[i, :j], A[j, :j])
            A[i][j] /= A[j][j]
        A[i][i] -= pk.dot(A[i, :i], A[i, :i])
        A[i][i] = pk.sqrt(A[i][i])

def cho_cp(A):
    A[0, 0] = cp.sqrt(A[0, 0])
    for i in range(1, A.shape[0]):
        for j in range(i):
            A[i, j] -= cp.dot(A[i, :j], A[j, :j])
            A[i, j] /= A[j, j]
        A[i, i] -= cp.dot(A[i, :i], A[i, :i])
        A[i, i] = cp.sqrt(A[i, i])

def run():
    N = 4
    A = cp.random.randn(N, N)
    A = A @ A.T
    A_np = cp.asnumpy(A)

    cho_cp(A)
    print(A)

    pk.set_default_space(pk.Cuda)
    B = pk.View((N, N), dtype=pk.float64, layout=pk.LayoutRight)
    np.copyto(B.data, A_np.data, casting="unsafe")

    cho_pk(B)
    print(B)


if __name__ == "__main__":
    run()
