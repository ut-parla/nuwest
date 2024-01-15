import pykokkos as pk

@pk.workunit()
def advection_kernel(
    tid,
    N, 
    x_ar, 
    v_ar, 
    E_ar, 
    R_ar,
    lhs_count,
    rhs_count,
    stride,
):
    # Looping and doing all the particles.
    for i in range(tid, Nc, stride):
        d_x: float = x_ar[i]
        d_v: float = v_ar[i]
        d_E: float = E_ar[i]
        d_R: float = R_ar[i]

        # Collision - each particle has 10% probability of collision, which flips velocity.
        if (d_R < 0.1):
            d_v = -1*d_v 
 
        # Advection in v-space.
        d_v += d_E

        # Advection in x-space.
        d_x += d_v

        # Reflective boundary condition - count boundary collisions with atomics.
        if (d_x > 1):
            pk.atomic_add(rhs_count, [0], 1)
            d_x = 1 - (d_x - 1)
            d_v = -d_v
        elif (d_x < 0):
            pk.atomic_add(lhs_count, [0], 1)
            d_x = -d_x
            d_v = -d_v       
 
        # Put data back into arrays.
        x_ar[i] = d_x
        v_ar[i] = d_v
        E_ar[i] = d_E 

def advect(
    N,
    x_ar,
    v_ar,
    E_ar,
    R_ar,
    lhs_count,
    rhs_count,
    threads_per_block,
    num_blocks,
):

    num_threads = num_blocks * threads_per_block
    
    # Launch PyKokkos kernel
    pk.parallel_for(
        num_threads,
        advection_kernel,
        N=N,
        x_ar=x_ar,
        v_ar=v_ar,
        E_ar=E_ar,
        R_ar=R_ar,
        lhs_count=lhs_count,
        rhs_count=rhs_count,
        stride=num_threads,
    )
