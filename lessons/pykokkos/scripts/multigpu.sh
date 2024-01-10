#!/bin/bash

name=mini_boltzmann_multigpu.py

rm -r __pycache__
rm -r pk_cpp
python "${name}" -g 1 -N 100000 -s 10
python "${name}" -g 2 -N 100000 -s 10
python "${name}" -g 3 -N 100000 -s 10
