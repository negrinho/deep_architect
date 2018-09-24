#!/bin/bash

mpiexec -np $1 python examples/tutorials/full_search/search.py -c mpi -n $1