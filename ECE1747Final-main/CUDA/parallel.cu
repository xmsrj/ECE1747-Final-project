#include<stdio.h>
#include <bits/stdc++.h>
#include<cuda.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

using namespace std;


void get_adj_matrix(float* graph, int n, float d, FILE *inputFilePtr ){

    //printf("Breakpoint reached!\n");
    if ( inputFilePtr == NULL )  {
        printf( "input.txt file failed to open." );
        return ;
    }

    //printf("Breakpoint reached!\n");

    int m, indexing;
    
    fscanf(inputFilePtr, "%d", &m);
    fscanf(inputFilePtr, "%d", &indexing);

    //printf("%d, %d\n", m, indexing);

    
    for(int i = 0; i< n ; i++){
        for(int j = 0; j< n; ++j){
            graph[i * n + j] = (1 - d)/float(n);
        }
    }

    //printf("Graph initialized!\n");

    while(m--){
        int source, destin;
        fscanf(inputFilePtr, "%d", &source);
        fscanf(inputFilePtr, "%d", &destin);

        //printf("%d, %d\n", source, destin);

        if (indexing == 0){
            graph[destin* n + source] += d* 1.0  ;
        }
        else{
            graph[(destin - 1)* n + source - 1] += d* 1.0;
        }
    }

}

__global__ void manage_adj_matrix(float* gpu_graph, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n){
        float sum = 0.0;

        for (int i = 0; i< n; ++i){
            sum += gpu_graph[i* n + id];
        }

        for (int i = 0; i < n; ++i){
            if (sum != 0.0){
                gpu_graph[i* n + id] /= sum;
            }
            else{
                gpu_graph[i* n + id] = (1/(float)n);
            }
        }
    }
}

__global__ void initialize_rank(float* gpu_r, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n){
        gpu_r[id] = (1/(float)n);
    }
}

__global__ void store_rank(float* gpu_r,float* gpu_r_last, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n){
        gpu_r_last[id] = gpu_r[id];
    }
}

__global__ void matmul(float* gpu_graph, float* gpu_r, float* gpu_r_last, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n){
        float sum = 0.0;

        for (int j = 0; j< n; ++j){
            sum += gpu_r_last[j] * gpu_graph[id* n + j];
        }

        gpu_r[id] = sum;
    }
}

__global__ void rank_diff(float* gpu_r,float* gpu_r_last, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n){
        gpu_r_last[id] = abs(gpu_r_last[id] - gpu_r[id]);
    }
}

__global__ void init_pair_array(pair<float, int>* gpu_r_nodes, float * gpu_r, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n){
        gpu_r_nodes[id].first = gpu_r[id];
        gpu_r_nodes[id].second = id + 1;
    }
}


void power_method(float *graph, float *r, int n, int nblocks, int BLOCKSIZE, int max_iter = 1000, float eps = 0.000001 ){
   
    float* r_last = (float*) malloc(n * sizeof(float));
    
    float* gpu_graph;
    cudaMalloc(&gpu_graph, sizeof(float)*n*n);
    cudaMemcpy(gpu_graph, graph, sizeof(float)*n*n, cudaMemcpyHostToDevice);

    float* gpu_r;
    cudaMalloc(&gpu_r, sizeof(float)*n);
    //cudaMemcpy(gpu_r, r, sizeof(float)*n, cudaMemcpyHostToDevice);

    float* gpu_r_last;
    cudaMalloc(&gpu_r_last, sizeof(float)*n);
    //cudaMemcpy(gpu_r_last, r_last, sizeof(float)*n, cudaMemcpyHostToDevice);



    initialize_rank<<<nblocks, BLOCKSIZE>>>(gpu_r, n);
    cudaDeviceSynchronize();



    while(max_iter--){

        store_rank<<<nblocks, BLOCKSIZE>>>(gpu_r, gpu_r_last, n);
        cudaDeviceSynchronize();

        matmul<<<nblocks, BLOCKSIZE>>>(gpu_graph, gpu_r, gpu_r_last, n);
        cudaDeviceSynchronize();
        
        rank_diff<<<nblocks, BLOCKSIZE>>>(gpu_r, gpu_r_last, n);
        cudaDeviceSynchronize();

        cudaMemcpy(r_last, gpu_r_last, n* sizeof(float), cudaMemcpyDeviceToHost);
        float result = thrust::reduce( r_last, r_last + n);

        if(result < eps){
            cudaMemcpy(r, gpu_r, n* sizeof(float), cudaMemcpyDeviceToHost);
            return;
        }
    }
    cudaMemcpy(r, gpu_r, n* sizeof(float), cudaMemcpyDeviceToHost);
    return;
}

void top_nodes(float* r, int n, int nblocks, int BLOCKSIZE, int count = 3){

    pair<float, int> *r_nodes = (pair<float, int> *) malloc ( n * sizeof (pair<float, int>) );
    pair<float, int> *gpu_r_nodes;

    cudaMalloc(&gpu_r_nodes, n * sizeof (pair<float, int>));

    float* gpu_r;
    cudaMalloc(&gpu_r, sizeof(float)*n);
    cudaMemcpy(gpu_r, r, sizeof(float)*n, cudaMemcpyHostToDevice);

    init_pair_array<<<nblocks, BLOCKSIZE>>>(gpu_r_nodes, gpu_r, n);

    cudaMemcpy(r_nodes, gpu_r_nodes, n * sizeof (pair<float, int>), cudaMemcpyDeviceToHost);

    thrust::sort(thrust::host, r_nodes, r_nodes + n);

    int rank =1;
    while(rank <= count){
        printf("Rank %d Node is %d\n", rank, r_nodes[n - rank].second);
        rank++;
    }
}

int main(int argc, char** argv){

    clock_t start, end;


    FILE *inputFilePtr;

    char * inputfile = argv[1];

    char * bsize = argv[2];


    int BLOCKSIZE = atoi(bsize);


    inputFilePtr = fopen(inputfile, "r");

    int n; 

    fscanf(inputFilePtr, "%lld", &n);

   // printf("%lld\n",n);

    int nblocks = ceil(float(n) / BLOCKSIZE);

    //printf("Breakpoint reached!\n");

    float* graph = (float*)malloc(n*n*sizeof(float));
    float* r = (float*) malloc(n * sizeof(float));

    float d = 0.85;

    //printf("%lld\n", n * n);

    get_adj_matrix(graph, n, d, inputFilePtr);

    //printf("Breakpoint reached!\n");

    float* gpu_graph;
    cudaMalloc(&gpu_graph, sizeof(float)*n*n);
    cudaMemcpy(gpu_graph, graph, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    
    start = clock();

    manage_adj_matrix<<<nblocks, BLOCKSIZE>>>(gpu_graph, n);
    cudaMemcpy(graph, gpu_graph, sizeof(float)*n*n, cudaMemcpyDeviceToHost);

    power_method(graph, r, n, nblocks, BLOCKSIZE );

    //top_nodes(r, n, nblocks, BLOCKSIZE);

    end = clock();

    printf("Time taken :%f (ms) for parallel implementation with %d nodes and %d threads per block.\n", 1000.0 * float(end - start) / CLOCKS_PER_SEC, n, BLOCKSIZE);
    return 0;
}