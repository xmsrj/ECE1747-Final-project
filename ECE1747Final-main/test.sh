#!/usr/bin/env bash

# ------------------ Set up common parameters ------------------

data_path=./data/new.txt
num_lines=10000000

# ------------------ Test for serial code ------------------

echo $'\n------------Testing serial code...------------\n'
./build/serial ${data_path} ${num_lines}
echo $'\n------------Done!------------\n'
# ------------------ Test for omp code ------------------

num_threads=8

echo $'\n------------Testing omp code...------------\n'
./build/omp ${data_path} ${num_threads} ${num_lines}
echo $'\n------------Done!------------\n'

# ------------------ Test for mpi code ------------------

num_nodes=4

echo $'\n------------Testing mpi code...------------\n'
mpirun -n ${num_nodes} ./build/mpi ${data_path} ${num_lines}
echo $'\n------------Done!------------\n'