# ECE1747Final

data collected from https://networkrepository.com/web.php

To compile serial/OpenMP/MPI code,

  ```
  mkdir build && cd build
  
  cmake ..
  
  make
  ```

Then you will find three executables serial, omp and mpi.

Run these executables by

```
  ./serial [path_to_data.txt]
  ./omp [path_to_data.txt]
  mpirun -n [number of processers] [path_to_data.txt]
```


To compile CUDA code,

``` nvcc parallel.cu -o cuda ```

and to run CUDA code,

```./cuda [path_to_data.txt] [number_of_threads_per_node]```
