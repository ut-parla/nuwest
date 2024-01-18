
import pykokkos as pk

@pk.workunit
def hello(i: int):
    pk.printf("Hello, World! from i = %d\n", i)

def main():
    pk.parallel_for(10, hello)

main()
