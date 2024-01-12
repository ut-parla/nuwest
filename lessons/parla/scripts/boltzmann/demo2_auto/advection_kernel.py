import math
import cupy as cp
import pykokkos as pk

@pk.workunit()

def pk_advection_kernel(
    tid,
    Nc, 
    d_x_ar,
    d_v_ar,
    d_E_ar,
    stride,
):
    # Looping and doing all the particles 
    for i in range(tid, Nc, stride):
        d_x: float = d_x_ar[i]
        d_v: float = d_v_ar[i]
        d_E: float = d_E_ar[i]
        
        # Advection in v-space
        d_v += d_E

        # Advection in x-space
        d_x += d_v

        # Reflective boundary condition
        if (d_x > 1):
            d_x = 1 - (d_x - 1)
            d_v = -d_v
        elif (d_x < 0):
            d_x = -d_x
            d_v = -d_v       
 
        # Put data back into arrays
        d_x_ar[i] = d_x
        d_v_ar[i] = d_v
        d_E_ar[i] = d_E 

def advect(
    Nc,
    d_x_ar,
    d_v_ar,
    d_E_ar,
    threads_per_block,
    num_blocks,
):

    num_threads = num_blocks * threads_per_block
    
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
