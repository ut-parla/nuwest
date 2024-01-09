import argparse

import numba

from parla import Parla, TaskSpace, spawn
from parla.cython.device_manager import cpu, gpu

try:
    from boltzmann import BoltzmannDSMC
except:
    from .boltzmann import BoltzmannDSMC


def main(restart_cyclenum, num_steps, name, opc, zero_dim, zerod_ef, null_coll, num_particles, gridpts, bnf, num_nodes, verbose, pyk, recomb, ecov, num_gpus):

    boltzmann = BoltzmannDSMC(zero_dim, zerod_ef, null_coll, num_steps, opc, name, restart_cyclenum, num_particles, gridpts, bnf, num_nodes, pyk, recomb, ecov, num_gpus)
    with Parla():    
        boltzmann.run(num_steps, name, zero_dim, verbose, restart_cyclenum)

def run(restart_cyclenum, num_steps, name, opc, zero_dim, zerod_ef, null_coll, num_particles, gridpts, bnf, num_nodes, verbose, pyk, recomb, no_warmup, ecov, num_gpus):

    main(restart_cyclenum, num_steps, name, opc, zero_dim, zerod_ef, null_coll, num_particles, gridpts, bnf, num_nodes, verbose, pyk, recomb, ecov, num_gpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--steps", type=int, default=400)
    parser.add_argument("-N", "--num_particles", type=int, default=200000)
    parser.add_argument("-g", "--gridpts", type=int, default=256)
    parser.add_argument("-bnf", type=int, default=4)
    parser.add_argument("-n", "--num_nodes", type=int, default=1)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--pykokkos", action="store_true")
    parser.add_argument("-r", "--recomb", action="store_true")
    parser.add_argument("-nw", "--no_warmup", action="store_true")
    parser.add_argument("-e", "--Ecov", type=float, default=5.0)
    parser.add_argument("-res", "--restart_cyclenum", type=int, default=0)
    parser.add_argument("-zero", "--zero_dim", action="store_true")
    parser.add_argument("-zdef", "--zerod_ef", type=float, default=0.0)
    parser.add_argument("-nc", "--null_coll", action="store_true")
    parser.add_argument("-name", "--data_name", type=str, default="")
    parser.add_argument("-opc", "--outputs_per_cycle", type=int, default=1)
    parser.add_argument("-ng", "--num_gpus", type=int, default=1)
    args = parser.parse_args()
    profile = run(args.restart_cyclenum, args.steps, args.data_name, args.outputs_per_cycle, args.zero_dim, args.zerod_ef, args.null_coll,
                  args.num_particles, args.gridpts, args.bnf, args.num_nodes, args.verbose, args.pykokkos, args.recomb, args.no_warmup, args.Ecov, args.num_gpus)
    from pprint import pprint

    pprint(profile)
