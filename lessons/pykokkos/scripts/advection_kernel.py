import numpy as np
import math
import pykokkos as pk

# pk.set_default_space(pk.OpenMP)


@pk.workunit()
# d_x_ar,
# d_v_ar,
# d_E_ar,
# d_x_ar=pk.ViewTypeInfo(space=pk.HostSpace),
# d_v_ar=pk.ViewTypeInfo(space=pk.HostSpace),
# d_E_ar=pk.ViewTypeInfo(space=pk.HostSpace),
# )
def pk_advection_kernel(
    tid: int,
    Nc: int,
    d_x_ar: pk.View1D[float],
    d_v_ar: pk.View1D[float],
    d_E_ar: pk.View1D[float],
    stride: int,
):
    # Looping and doing all the particles
    for i in range(tid, Nc, stride):
        if i == 0:
            printf("hi from kernel \n")
        d_x: float = d_x_ar[i]
        d_v: float = d_v_ar[i]
        d_E: float = d_E_ar[i]

        # Advection in v-space
        d_v += d_E

        # Advection in x-space
        d_x += d_v

        # Reflective boundary condition
        if d_x > 1:
            d_x = 1 - (d_x - 1)
            d_v = -d_v
        elif d_x < 0:
            d_x = -d_x
            d_v = -d_v

        # Put data back into arrays
        d_x_ar[i] = d_x
        d_v_ar[i] = d_v
        d_E_ar[i] = d_E


def advect(Nc, d_x_ar, d_v_ar, d_E_ar, threads_per_block, num_blocks, i):
    num_threads: int = num_blocks * threads_per_block
    print("Launch Kernel", i, flush=True)
    # Launch PyKokkos kernel
    pk.parallel_for(
        num_threads,
        pk_advection_kernel,
        Nc=Nc,
        d_x_ar=d_x_ar,
        d_v_ar=d_v_ar,
        d_E_ar=d_E_ar,
        stride=num_threads,
    )
    print("Finish Kernel", i, flush=True)
