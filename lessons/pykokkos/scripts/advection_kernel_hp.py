import pykokkos as pk

@pk.workunit()
def advection_kernel(
    team_member,
    N, 
    x_ar, 
    v_ar, 
    E_ar, 
    R_ar,
):
    j: int = team_member.league_rank()
    k: int = team_member.team_size()

    def inner(i: int):
        # Looping and doing all the particles.
        ix: int = j * k + i
        d_x: float = x_ar[ix]
        d_v: float = v_ar[ix]
        d_E: float = E_ar[ix]
        d_R: float = R_ar[ix]
        
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
        x_ar[ix] = d_x
        v_ar[ix] = d_v
        E_ar[ix] = d_E
    pk.parallel_for(pk.TeamThreadRange(team_member, k), inner)

def advect(
    N,
    x_ar,
    v_ar,
    E_ar,
    R_ar,
    threads_per_block,
    num_blocks,
):
    # Launch PyKokkos kernel.
    pk.parallel_for(
        pk.TeamPolicy(num_blocks, 2),
        advection_kernel,
        N=N,
        x_ar=x_ar,
        v_ar=v_ar,
        E_ar=E_ar,
        R_ar=R_ar,
    )
