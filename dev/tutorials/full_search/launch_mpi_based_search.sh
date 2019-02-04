#!/bin/bash

mpiexec -np $1 python tutorials/full_search/search.py -c mpi -n $1