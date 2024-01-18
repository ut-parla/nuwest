
import pykokkos as pk
import numpy as np

@pk.workunit
def work(wid, acc, a):
    acc += a[wid]

def main():
    N = 10
    a = np.random.randint(100, size=(N))
    print(a)
    total = pk.parallel_reduce("work", N, work, a=a)
    print(total)

main()
