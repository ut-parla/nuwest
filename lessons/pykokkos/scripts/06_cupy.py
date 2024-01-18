
import cupy as cp
import pykokkos as pk

@pk.workunit
def work(wid, a):
    a[wid] = a[wid] + 1

def main():
    N = 10
    a = cp.ones(N)
    pk.set_default_space(pk.Cuda)
    pk.parallel_for("work", 10, work, a=a)
    print(a)

main()
