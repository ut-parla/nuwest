import pykokkos as pk

@pk.workunit()
def advection_kernel(
    team_member,
    Nc, 
    d_x_ar, 
    d_v_ar, 
    d_E_ar, 
    d_R_ar,
    stride,
):
    j: int = team_member.league_rank()

    def inner(i: int):
        # Looping and doing all the particles.
        d_x: float = d_x_ar[i]
        d_v: float = d_v_ar[i]
        d_E: float = d_E_ar[i]
        d_R: float = d_R_ar[i]
        
        # Collision - each particle has 10% probability of collision, which flips velocity.
        if (d_R < 0.1):
            d_v = -1*d_v 
        
        # Advection in v-space.
        d_v += d_E
        
        # Advection in x-space.
        d_x += d_v
        
        # Reflective boundary condition.
        if (d_x > 1):
            d_x = 1 - (d_x - 1)
            d_v = -d_v
        elif (d_x < 0):
            d_x = -d_x
            d_v = -d_v       
        
        # Put data back into arrays.
        d_x_ar[i] = d_x
        d_v_ar[i] = d_v
        d_E_ar[i] = d_E
    pk.parallel_for(pk.TeamThreadRange(team_member, stride), inner)

def advect(
    Nc,
    d_x_ar,
    d_v_ar,
    d_E_ar,
    d_R_ar,
    threads_per_block,
    num_blocks,
):
    # Launch PyKokkos kernel.
    pk.parallel_for(
        pk.TeamPolicy(num_blocks, pk.AUTO),
        advection_kernel,
        Nc=Nc,
        d_x_ar=d_x_ar,
        d_v_ar=d_v_ar,
        d_E_ar=d_E_ar,
        d_R_ar=d_R_ar,
        stride=threads_per_block,
    )
