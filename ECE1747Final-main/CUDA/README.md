# PageRank accelerated by CUDA
Parallelizing Google's PageRank algorithm in C++ with CUDA framework on GPU. =


- [Sequential Code (C++)](./sequential.cpp)
- [Parallel Code (C++ with CUDA)](.parallel.cu)
- [Datasets (File name indicates number of nodes)](../data)

# Examples Usage:

Sequential

g++ sequential.cpp
./a.out 18000.txt

CUDA parallization
nvcc parallel.cu
./a.out ./data/18000.txt 32 / ./a.out ./data/18000.txt 64
The last argument is the number of threads per node 

