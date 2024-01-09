#!/bin/bash

rm -r __pycache__
rm -r pk_cpp
python mini_boltzmann_cpu.py -N 100000 -s 10
