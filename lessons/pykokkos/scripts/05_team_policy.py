
import numpy as np
import pykokkos as pk

@pk.workunit
def work(team_member, view):
    j: int = team_member.league_rank()
    k: int = team_member.team_size()

    def inner(i: int):
        view[j * k + i] = view[j * k + i] + 1

    pk.parallel_for(pk.TeamThreadRange(team_member, k), inner)

def main():
    pk.set_default_space(pk.OpenMP)
    a = np.zeros(100)
    pk.parallel_for("work", pk.TeamPolicy(50, 2), work, view=a)
    print(a)

main()
