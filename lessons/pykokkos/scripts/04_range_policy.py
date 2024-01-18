
import pykokkos as pk
import numpy as np

@pk.workunit
def work(wid, a):
    a[wid] += 1

def main():
    N = 10
    a = np.random.randint(100, size=(N))
    print(a)

    pk.parallel_for("work", pk.RangePolicy(0, N), work, a=a)
    # OR
    # pk.parallel_for("work", N, work, a=a)
    print(a)

main()
