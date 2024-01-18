
import pykokkos as pk
import numpy as np

@pk.workunit
def work(wid, acc, final, a):
    acc += a[wid]
    if final:
        a[wid] = acc

def main():
    N = 10
    a = np.random.randint(100, size=(N))
    print(a)

    pk.parallel_scan("work", N, work, a=a)
    print(a)

main()
