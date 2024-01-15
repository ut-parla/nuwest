from cProfile import Profile
from pstats import SortKey, Stats

import pykokkos as pk


@pk.workunit
def stencil(tid: int, inp: pk.View3D[float], out: pk.View3D[float], p: int):
    N: int = inp.extent(1)
    for i in range(p):
        j_x: int = (tid // (N - 2)) + 1
        j_y: int = (tid % (N - 2)) + 1

        out[i][j_x][j_y] += inp[i][j_x][j_y-1] * -0.5 + inp[i][j_x-1][j_y] * -0.5 + inp[i][j_x+1][j_y] * 0.5 + inp[i][j_x][j_y+1] * 0.5

@pk.workunit
def copy_1D(tid: int, u: pk.View3D[float], f: pk.View3D[float], p: int):
    N: int = u.extent(1)
    j_x: int = tid // N
    j_y: int = tid % N

    for i in range(p):
        u[i][j_x][j_y] = f[i][j_x][j_y]

@pk.workunit
def inner_kernel(tid: int, u: pk.View3D[float], B: pk.View2D[float], v: pk.View1D[float], f: pk.View3D[float], w: pk.View3D[float], gemv_out: pk.View3D[float]):
    p: int = u.extent(0)
    N: int = u.extent(1)

    j_x: int = tid // N
    j_y: int = tid % N

    for i in range(p):
        w[j_x][j_y][i] = u[i][j_x][j_y]

    for i in range(p):
        temp: float = 0
        for j in range(p):
            temp += B[i][j] * w[j_x][j_y][j]

        gemv_out[j_x][j_y][i] = temp

    dot_out: float = 0
    for i in range(p):
        dot_out += v[i] * w[j_x][j_y][i]

    for i in range(p):
        gemv_out[j_x][j_y][i] *= dot_out

    for i in range(p):
        f[i][j_x][j_y] += gemv_out[j_x][j_y][i]

def run():
    N = 100
    max_iterations = 10
    p = 100
    u_views = pk.View((p, N, N))
    f_views = pk.View((p, N, N))
    B = pk.View((p, p))
    v = pk.View((p,))

    for _ in range(max_iterations):
        pk.parallel_for("stencil", (N - 2) * (N - 2), stencil, inp=u_views, out=f_views, p=p)

        w_views = pk.View((N, N, p))
        gemv_out = pk.View((N, N, p)) # temporary storage to hold the output of the gemv

        pk.parallel_for("second kernel", N * N, inner_kernel, u=u_views, B=B, v=v, f=f_views, w=w_views, gemv_out=gemv_out)
        pk.parallel_for("copy", N * N, copy_1D, u=u_views, f=f_views, p=p)

if __name__ == "__main__":
    run()
