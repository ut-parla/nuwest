import cupy as cp
import pykokkos as pk

# Code taken from https://github.com/spcl/npbench and then revised to
# show cp vs pk usage.

def kernel(A):
    A[0][0] = pk.sqrt(A[0][0])
    for i in range(1, A.shape[0]):
        for j in range(i):
            A[i][j] -= pk.dot(A[i, :j], A[j, :j])
            A[i][j] /= A[j][j]
        A[i][i] -= pk.dot(A[i, :i], A[i, :i])
        A[i][i] = pk.sqrt(A[i][i])

def kernel_cp(A):
    A[0, 0] = cp.sqrt(A[0, 0])
    for i in range(1, A.shape[0]):
        for j in range(i):
            A[i, j] -= cp.dot(A[i, :j], A[j, :j])
            A[i, j] /= A[j, j]
        A[i, i] -= cp.dot(A[i, :i], A[i, :i])
        A[i, i] = cp.sqrt(A[i, i])

def run():
    arr = cp.ones((10, 10))
    kernel_cp(arr)
    print(arr)

    pk_arr = pk.View((10, 10))
    pk_arr.fill(1)
    kernel(pk_arr)
    print(pk_arr)


if __name__ == "__main__":
    run()
