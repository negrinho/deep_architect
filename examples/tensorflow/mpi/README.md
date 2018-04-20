In order to use this run script, you must have an implementation of MPI installed on your system.
Installation instructions and other documentation can be found here: http://www.mpich.org/documentation/guides/

Next, you must install mpi4py: http://mpi4py.scipy.org/docs/usrman/install.html

Finally, in order to run, simply execute the command `mpiexec -n NUMPROCESSES python mpi_run.py -c CONFIG_NAME`
where NUMPROCESSES is equal to the number of workers you want + 1.
