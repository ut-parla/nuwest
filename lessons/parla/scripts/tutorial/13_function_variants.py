from typing import Callable, Optional
from time import perf_counter, sleep

# Handle for Parla runtime
from parla import Parla

# Spawning  tasks
from parla.tasks import spawn, TaskSpace, specialize
from parla.devices import cpu, gpu

from time import perf_counter, sleep

from helper import run

from parla.array import clone_here
import numpy as np
import cupy as cp


@specialize
def function(A: np.ndarray):
    print("Running Function's Default Implementation", flush=True)
    return np.linalg.eigh(A)


@function.variant(gpu)
def function_gpu(A: cp.ndarray):
    print("Running Function's GPU Implementation", flush=True)
    return cp.linalg.eigh(A)


def specialization_example():
    A = np.random.rand(1000, 1000)
    B = np.copy(A)
    T = TaskSpace("T")

    @spawn(T[0], placement=cpu)
    def t1():
        print("Running CPU Task", flush=True)
        A_local = clone_here(A)
        C = function(A_local)
        print("Completed CPU Task", flush=True)

    @spawn(T[1], [T[0]], placement=gpu)
    def t2():
        print("Running GPU Task", flush=True)
        B_local = clone_here(B)
        C = function(B_local)
        print("Completed GPU Task", flush=True)

    return T


run(specialization_example)
