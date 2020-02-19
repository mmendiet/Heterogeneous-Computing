#include <iostream>
#include <cuda.h>
#include <ctime>
#include <chrono>
#include <unistd.h>
#include <bits/stdc++.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#define TILE_DIM 32

/*
void vector_add_cpu(int* a, int* b, int* c, int size);
void vector_add_cuda(int* a, int* b, int* c, int size, int TB_size, float& time, float& time_with_memcpy);
void print_array(int* array, int size);
void stencil_cpu(int* in, int* out, int size, int radius);
void stencil_cuda(int* in, int* out, int size, int radius, int TB_size, float& time, float& time_with_memcpy);
void stencil_cuda_shared(int* in, int* out, int size, int radius, int TB_size, float& time, float& time_with_memcpy);
*/
void matrix_naive_cuda(int *matA, int *matB, int *matC, int n, float& time, float& time_memcpy);
void matrix_coal_cuda(int *matA, int *matB, int *matC, int n, float& time, float& time_memcpy);
void matrix_tiling_cuda(int *matA, int *matB, int *matC, int n, float& time, float& time_memcpy);
void matrix_tiling_coal_cuda(int *matA, int *matB, int *matC, int n, float& time, float& time_memcpy);
void transpose_matrix(int* mat, int* trans_mat, int n);
void print_matrix(int* matrix, int size);

static void show_usage(std::string name)
{
    std::cerr << "Usage: vector_add <option(s)> "
              << "Options:\n"
              << "\t-h,--help\t\tShow this help message\n"
              << "\t--N, int\tSpecify the vector size"
              << "\t--output, filename.txt\tSpecify the output file name and path"
              << "\t--job_size, int\tSpecify the job size"
              << "\t--TB_size, int\tSpecify the Thread Block size"
              << std::endl;
}
/*
__global__ void vector_add(int *a, int *b, int *c, int size){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        c[tid] = a[tid] + b[tid];
    }

}

__global__ void stencil_gpu(int *in, int* out, int size, int radius){
    int gindex =  threadIdx.x + blockIdx.x * blockDim.x;
    int result = 0;
    for (int i = -radius; i <= radius; i++){
        if( ((i<0) && (gindex < radius)) || ((i>0) && (gindex > (size - radius)) )){
            result += 0;
        }
        else{
            result += in[gindex + i];
        }
    }
    out[gindex] = result;
}


__global__ void stencil_shared_r2(int *in, int *out, int block_size){
    __shared__ int temp[132];
    int radius = 2;
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + radius;

    temp[lindex] = in[gindex];
    if (threadIdx.x < radius) {
        if (blockIdx.x == 0){
            temp[lindex - radius] = 0;
            temp[lindex + block_size] = in[gindex + block_size];
        }
        else if(blockIdx.x == (gridDim.x - 1)){
            temp[lindex - radius] = in[gindex-radius];
            temp[lindex + block_size] = 0;
        }
        else{
            temp[lindex - radius] = in[gindex-radius];
            temp[lindex + block_size] = in[gindex + block_size];
        }
    }
    __syncthreads();

    int result = 0;
    for (int offset = -radius; offset <= radius; offset++){
        result += temp[lindex + offset];
    }
}

__global__ void stencil_shared_r4(int *in, int *out, int block_size){
    __shared__ int temp[136];
    int radius=4;
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + radius;

    temp[lindex] = in[gindex];
    if (threadIdx.x < radius) {
        if (blockIdx.x == 0){
            temp[lindex - radius] = 0;
            temp[lindex + block_size] = in[gindex + block_size];
        }
        else if(blockIdx.x == (gridDim.x - 1)){
            temp[lindex - radius] = in[gindex-radius];
            temp[lindex + block_size] = 0;
        }
        else{
            temp[lindex - radius] = in[gindex-radius];
            temp[lindex + block_size] = in[gindex + block_size];
        }
    }
    __syncthreads();

    int result = 0;
    for (int offset = -radius; offset <= radius; offset++){
        result += temp[lindex + offset];
    }
}

__global__ void stencil_shared_r8(int *in, int *out, int block_size){
    __shared__ int temp[144];
    int radius = 8;
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + radius;

    temp[lindex] = in[gindex];
    if (threadIdx.x < radius) {
        if (blockIdx.x == 0){
            temp[lindex - radius] = 0;
            temp[lindex + block_size] = in[gindex + block_size];
        }
        else if(blockIdx.x == (gridDim.x - 1)){
            temp[lindex - radius] = in[gindex-radius];
            temp[lindex + block_size] = 0;
        }
        else{
            temp[lindex - radius] = in[gindex-radius];
            temp[lindex + block_size] = in[gindex + block_size];
        }
    }
    __syncthreads();

    int result = 0;
    for (int offset = -radius; offset <= radius; offset++){
        result += temp[lindex + offset];
    }
}

__global__ void stencil_shared_r16(int *in, int *out, int block_size){
    __shared__ int temp[160];
    int radius = 16;
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + radius;

    temp[lindex] = in[gindex];
    if (threadIdx.x < radius) {
        if (blockIdx.x == 0){
            temp[lindex - radius] = 0;
            temp[lindex + block_size] = in[gindex + block_size];
        }
        else if(blockIdx.x == (gridDim.x - 1)){
            temp[lindex - radius] = in[gindex-radius];
            temp[lindex + block_size] = 0;
        }
        else{
            temp[lindex - radius] = in[gindex-radius];
            temp[lindex + block_size] = in[gindex + block_size];
        }
    }
    __syncthreads();

    int result = 0;
    for (int offset = -radius; offset <= radius; offset++){
        result += temp[lindex + offset];
    }
}
*/
__global__ void matrix_naive(int *matA, int *matB, int *matC, int n){

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int result = 0;
    
    if (row < n && col < n){
        for(int i=0; i < n; i++){
            result += matA[row*n+i] * matB[i*n+col];
        }
    }
    matC[row*n+col] = result;
}

__global__ void matrix_coalescing(int *matA, int *matB, int *matC, int n){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int result = 0;
    
    if (row < n && col < n){
        for(int i=0; i < n; i++){
            result += matA[row+n*i] * matB[col+n*i];
        }
    }
    matC[row*n+col] = result;
}

__global__ void matrix_tiling(int *matA, int *matB, int *matC, int n){
    __shared__ int ATile[TILE_DIM][TILE_DIM];
    __shared__ int BTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int result = 0;

    for (int k = 0; k < (((n-1)/TILE_DIM)+1); k++){
        if (row < n && (threadIdx.x + (k*TILE_DIM)) < n){
            ATile[threadIdx.y][threadIdx.x] = matA[row*n+k*TILE_DIM+threadIdx.x];
        }
        else{
            ATile[threadIdx.y][threadIdx.x] = 0;
        }
        if (col < n && (threadIdx.y + (k*TILE_DIM)) < n){
            BTile[threadIdx.y][threadIdx.x] = matB[(k*TILE_DIM+threadIdx.y)*n+col];
        }
        else{
            BTile[threadIdx.y][threadIdx.x] = 0;
        }
//        ATile[threadIdx.y][threadIdx.x] = matA[row*TILE_DIM+threadIdx.x];
//        BTile[threadIdx.y][threadIdx.x] = matB[threadIdx.y*n+col];
        __syncthreads();

        if (row < n && col < n){
            for (int i = 0; i < TILE_DIM; i++){
                result += ATile[threadIdx.y][i]*BTile[i][threadIdx.x];
            }
            matC[row*n+col] = result;
        }
    }
    
}

__global__ void matrix_tiling_coalescing(int *matA, int *matB, int *matC, int n){
    __shared__ int ATile[TILE_DIM][TILE_DIM];
    __shared__ int BTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int result = 0;

    if (row < n && col < n){
        ATile[threadIdx.y][threadIdx.x] = matA[row+TILE_DIM*threadIdx.x];
        BTile[threadIdx.y][threadIdx.x] = matB[threadIdx.y*n+col];
        __syncthreads();

        for (int i = 0; i < TILE_DIM; i++){
            result += ATile[threadIdx.y][i]*BTile[i][threadIdx.x];
        }
    }
    matC[row*n+col] = result;
}

int main(int argc, char* argv[]){

    std::string outfile;
    int vector_size, job_size, TB_size, radius;
    if(std::string(argv[1]) == "-h"){
        show_usage(argv[0]);
        return 0;
    }
    if(argc < 2){
        std::cerr << "Not enough arguments." << std::endl;
        return 1;
    }
    for (size_t i = 1; i < argc; i++)
    {
        if(std::string(argv[i]) == "--output"){
            if ((i + 1)< argc){
                outfile = argv[++i];
            }
            else{
                std::cerr << "--output option requires a path as an argument." << std::endl;
                return 1;
            }
        }
    }
    
    int *matA_100, *matB_100, *matC_100, *matA_1k, *matB_1k, *matC_1k, *matA_10k, *matB_10k, *matC_10k;
    float naive_100, naive_1k, naive_10k, coal_100, coal_1k, coal_10k, shared_100, shared_1k, shared_10k, shared_coal_100, shared_coal_1k, shared_coal_10k;
    float naive_100_cpy, naive_1k_cpy, naive_10k_cpy, coal_100_cpy, coal_1k_cpy, coal_10k_cpy, shared_100_cpy, shared_1k_cpy, shared_10k_cpy, shared_coal_100_cpy, shared_coal_1k_cpy, shared_coal_10k_cpy;

    matA_100 = (int*) malloc(100*100*sizeof(int));
    matB_100 = (int*) malloc(100*100*sizeof(int));
    matC_100 = (int*) malloc(100*100*sizeof(int));
    matA_1k = (int*) malloc(1000*1000*sizeof(int));
    matB_1k = (int*) malloc(1000*1000*sizeof(int));
    matC_1k = (int*) malloc(1000*1000*sizeof(int));
    matA_10k = (int*) malloc(10000*10000*sizeof(int));
    matB_10k = (int*) malloc(10000*10000*sizeof(int));
    matC_10k = (int*) malloc(10000*10000*sizeof(int));

    for(int i=0; i < 100; i++){
        for(int j=0; j < 100; j++){
            matA_100[i*100+j] = 1;
            matB_100[i*100+j] = 2;
            matC_100[i*100+j] = 0;
        }
    }
    for(int i=0; i < 1000; i++){
        for(int j=0; j < 1000; j++){
            matA_1k[i*1000+j] = 1;
            matB_1k[i*1000+j] = 2;
            matC_1k[i*1000+j] = 0;
        }
    }
    for(int i=0; i < 10000; i++){
        for(int j=0; j < 10000; j++){
            matA_10k[i*10000+j] = 1;
            matB_10k[i*10000+j] = 2;
            matC_10k[i*10000+j] = 0;
        }
    }

    matrix_naive_cuda(matA_100, matB_100, matC_100, 100, naive_100, naive_100_cpy);
    matrix_naive_cuda(matA_1k, matB_1k, matC_1k, 1000, naive_1k, naive_1k_cpy);
    matrix_naive_cuda(matA_10k, matB_10k, matC_10k, 10000, naive_10k, naive_10k_cpy);
    matrix_coal_cuda(matA_100, matB_100, matC_100, 100, coal_100, coal_100_cpy);
    matrix_coal_cuda(matA_1k, matB_1k, matC_1k, 1000, coal_1k, coal_1k_cpy);
    matrix_coal_cuda(matA_10k, matB_10k, matC_10k, 10000, coal_10k, coal_10k_cpy);
    matrix_tiling_cuda(matA_100, matB_100, matC_100, 100, shared_100, shared_100_cpy);
    matrix_tiling_cuda(matA_1k, matB_1k, matC_1k, 1000, shared_1k, shared_1k_cpy);
    matrix_tiling_cuda(matA_10k, matB_10k, matC_10k, 10000, shared_10k, shared_10k_cpy);
    matrix_tiling_coal_cuda(matA_100, matB_100, matC_100, 100, shared_coal_100, shared_coal_100_cpy);
    matrix_tiling_coal_cuda(matA_1k, matB_1k, matC_1k, 1000, shared_coal_1k, shared_coal_1k_cpy);
    matrix_tiling_coal_cuda(matA_10k, matB_10k, matC_10k, 10000, shared_coal_10k, shared_coal_10k_cpy);

    std::ofstream myfile;
    myfile.open (outfile);
    myfile << "Implementation,N=100,N=1k,N=10k,N=100k,N=1M\n";
    myfile << "Naive," << naive_100 << "," << naive_1k << ","<< naive_10k << "\n";//","<< naive_100k << ","<< naive_1M << "\n";
    myfile << "Naive w/ Memory Coalescing," << coal_100 << "," << coal_1k << ","<< coal_10k << "\n";//","<< coal_100k << ","<< coal_1M << "\n";
    myfile << "Tiling using Shared Memory," << shared_100 << "," << shared_1k << ","<< shared_10k << "\n";//","<< shared_100k << ","<< shared_1M << "\n";
    myfile << "Tiling using Shared Memory with Memory Coalescing," << shared_coal_100 << "," << shared_coal_1k << ","<< shared_coal_10k << "\n";//","<< shared_coal_100k << ","<< shared_coal_1M << "\n";
    myfile << "Naive including memcpy," << naive_100_cpy << "," << naive_1k_cpy << ","<< naive_10k_cpy << "\n";//","<< naive_100k_cpy << ","<< naive_1M_cpy << "\n";
    myfile << "Naive w/ Memory Coalescing and memcpy," << coal_100_cpy << "," << coal_1k_cpy << ","<< coal_10k_cpy << "\n";//","<< coal_100k << ","<< coal_1M << "\n";
    myfile << "Tiling using Shared Memory and memcpy," << shared_100_cpy << "," << shared_1k_cpy << ","<< shared_10k_cpy << "\n";//","<< shared_100k << ","<< shared_1M << "\n";
    myfile << "Tiling using Shared Memory with Memory Coalescing and memcpy," << shared_coal_100_cpy << "," << shared_coal_1k_cpy << ","<< shared_coal_10k_cpy << "\n";//","<< shared_coal_100k << ","<< shared_coal_1M << "\n";
    
    myfile.close();

    return 0;
}

void matrix_naive_cuda(int *matA, int *matB, int *matC, int n, float& time, float& time_memcpy){
    int *d_matA, *d_matB, *d_matC;
    int mem_size = n*n*sizeof(int);
    cudaStream_t s[2];
    cudaEvent_t start[2], stop[2];

    cudaEventCreate(&start[0]);
    cudaEventCreate(&stop[0]);
    cudaEventCreate(&start[1]);
    cudaEventCreate(&stop[1]);
    cudaStreamCreate(&s[0]);
    cudaStreamCreate(&s[1]);

    cudaMalloc((void **)&d_matA, mem_size);
    cudaMalloc((void **)&d_matB, mem_size);
    cudaMalloc((void **)&d_matC, mem_size);

    dim3 threadsPerBlock(16,16);
    int gridD = ceil(double(n)/16.0);
    dim3 blocksPerGrid(gridD, gridD);


    cudaEventRecord(start[0],s[0]);
    cudaMemcpy(d_matA, matA, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, matB, mem_size, cudaMemcpyHostToDevice);

    cudaEventRecord(start[1],s[1]);
    matrix_naive<<<blocksPerGrid,threadsPerBlock>>>(d_matA, d_matB, d_matC, n);

    cudaEventRecord(stop[1],s[1]);
    cudaEventSynchronize(stop[1]);
    cudaMemcpy(matC, d_matC, mem_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop[0],s[0]);
    cudaEventSynchronize(stop[0]);

    cudaEventElapsedTime(&time, start[1], stop[1]);
    cudaEventElapsedTime(&time_memcpy, start[0], stop[0]);
    cudaEventDestroy(start[0]);
    cudaEventDestroy(stop[0]);
    cudaEventDestroy(start[1]);
    cudaEventDestroy(stop[1]);
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);
}

void matrix_coal_cuda(int *matA, int *matB, int *matC, int n, float& time, float& time_memcpy){
    int *d_matA, *d_matB, *d_matC;
    int mem_size = n*n*sizeof(int);
    int *trans_matA;
    cudaStream_t s[2];
    cudaEvent_t start[2], stop[2];

    trans_matA = (int*) malloc(n*n*sizeof(int));
    transpose_matrix(matA, trans_matA, n);
    
    cudaEventCreate(&start[0]);
    cudaEventCreate(&stop[0]);
    cudaEventCreate(&start[1]);
    cudaEventCreate(&stop[1]);
    cudaStreamCreate(&s[0]);
    cudaStreamCreate(&s[1]);

    cudaMalloc((void **)&d_matA, mem_size);
    cudaMalloc((void **)&d_matB, mem_size);
    cudaMalloc((void **)&d_matC, mem_size);

    dim3 threadsPerBlock(16,16);
    int gridD = ceil(double(n)/16.0);
    dim3 blocksPerGrid(gridD, gridD);


    cudaEventRecord(start[0],s[0]);
    cudaMemcpy(d_matA, trans_matA, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, matB, mem_size, cudaMemcpyHostToDevice);

    cudaEventRecord(start[1],s[1]);
    matrix_coalescing<<<blocksPerGrid,threadsPerBlock>>>(d_matA, d_matB, d_matC, n);

    cudaEventRecord(stop[1],s[1]);
    cudaEventSynchronize(stop[1]);
    cudaMemcpy(matC, d_matC, mem_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop[0],s[0]);
    cudaEventSynchronize(stop[0]);

    cudaEventElapsedTime(&time, start[1], stop[1]);
    cudaEventElapsedTime(&time_memcpy, start[0], stop[0]);
    cudaEventDestroy(start[0]);
    cudaEventDestroy(stop[0]);
    cudaEventDestroy(start[1]);
    cudaEventDestroy(stop[1]);
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);
}

void matrix_tiling_cuda(int *matA, int *matB, int *matC, int n, float& time, float& time_memcpy){
    int *d_matA, *d_matB, *d_matC;
    int mem_size = n*n*sizeof(int);
    cudaStream_t s[2];
    cudaEvent_t start[2], stop[2];

    cudaEventCreate(&start[0]);
    cudaEventCreate(&stop[0]);
    cudaEventCreate(&start[1]);
    cudaEventCreate(&stop[1]);
    cudaStreamCreate(&s[0]);
    cudaStreamCreate(&s[1]);

    cudaMalloc((void **)&d_matA, mem_size);
    cudaMalloc((void **)&d_matB, mem_size);
    cudaMalloc((void **)&d_matC, mem_size);

    dim3 threadsPerBlock(16,16);
    int gridD = ceil(double(n)/16.0);
    dim3 blocksPerGrid(gridD, gridD);


    cudaEventRecord(start[0],s[0]);
    cudaMemcpy(d_matA, matA, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, matB, mem_size, cudaMemcpyHostToDevice);

    cudaEventRecord(start[1],s[1]);
    matrix_tiling<<<blocksPerGrid,threadsPerBlock>>>(d_matA, d_matB, d_matC, n);

    cudaEventRecord(stop[1],s[1]);
    cudaEventSynchronize(stop[1]);
    cudaMemcpy(matC, d_matC, mem_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop[0],s[0]);
    cudaEventSynchronize(stop[0]);

    cudaEventElapsedTime(&time, start[1], stop[1]);
    cudaEventElapsedTime(&time_memcpy, start[0], stop[0]);
    cudaEventDestroy(start[0]);
    cudaEventDestroy(stop[0]);
    cudaEventDestroy(start[1]);
    cudaEventDestroy(stop[1]);
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);
}

void matrix_tiling_coal_cuda(int *matA, int *matB, int *matC, int n, float& time, float& time_memcpy){
    int *d_matA, *d_matB, *d_matC;
    int mem_size = n*n*sizeof(int);
    int *trans_matA;
    cudaStream_t s[2];
    cudaEvent_t start[2], stop[2];

    trans_matA = (int*) malloc(n*n*sizeof(int));
    transpose_matrix(matA, trans_matA, n);
    
    cudaEventCreate(&start[0]);
    cudaEventCreate(&stop[0]);
    cudaEventCreate(&start[1]);
    cudaEventCreate(&stop[1]);
    cudaStreamCreate(&s[0]);
    cudaStreamCreate(&s[1]);

    cudaMalloc((void **)&d_matA, mem_size);
    cudaMalloc((void **)&d_matB, mem_size);
    cudaMalloc((void **)&d_matC, mem_size);

    dim3 threadsPerBlock(16,16);
    int gridD = ceil(double(n)/16.0);
    dim3 blocksPerGrid(gridD, gridD);


    cudaEventRecord(start[0],s[0]);
    cudaMemcpy(d_matA, trans_matA, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, matB, mem_size, cudaMemcpyHostToDevice);

    cudaEventRecord(start[1],s[1]);
    matrix_tiling_coalescing<<<blocksPerGrid,threadsPerBlock>>>(d_matA, d_matB, d_matC, n);

    cudaEventRecord(stop[1],s[1]);
    cudaEventSynchronize(stop[1]);
    cudaMemcpy(matC, d_matC, mem_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop[0],s[0]);
    cudaEventSynchronize(stop[0]);

    cudaEventElapsedTime(&time, start[1], stop[1]);
    cudaEventElapsedTime(&time_memcpy, start[0], stop[0]);
    cudaEventDestroy(start[0]);
    cudaEventDestroy(stop[0]);
    cudaEventDestroy(start[1]);
    cudaEventDestroy(stop[1]);
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);
}

void transpose_matrix(int* mat, int* trans_mat, int n){
    for(int i=0; i<n; i++){
        for(int j=0;j<n;j++){
            trans_mat[i*n+j] = mat[j*n+i];
        }
    }
    //print_matrix(trans_mat, n);
} 
/*
void vector_add_cpu(int* a, int* b, int* c, int size){
    for(int i = 0; i < size; i++){
        c[i] = a[i] + b[i];
    }

}

void vector_add_cuda(int* a, int* b, int* c, int size, int TB_size, float& time, float& time_with_memcpy){
    int *d_a, *d_b, *d_c;
    int mem_size = size*sizeof(int);
    cudaStream_t s[2];
    cudaEvent_t start[2], stop[2];
    //float time, time_with_memcpy;

    cudaEventCreate(&start[0]);
    cudaEventCreate(&stop[0]);
    cudaEventCreate(&start[1]);
    cudaEventCreate(&stop[1]);
    cudaStreamCreate(&s[0]);
    cudaStreamCreate(&s[1]);

    cudaMalloc((void **)&d_a, mem_size);
    cudaMalloc((void **)&d_b, mem_size);
    cudaMalloc((void **)&d_c, mem_size);

    cudaEventRecord(start[0],s[0]);
    cudaMemcpy(d_a, a, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, mem_size, cudaMemcpyHostToDevice);

    cudaEventRecord(start[1],s[1]);
    int M = TB_size;
    int N = (size + TB_size -1)/TB_size;
    vector_add<<<N,M>>>(d_a, d_b, d_c, size);

    cudaEventRecord(stop[1],s[1]);
    cudaEventSynchronize(stop[1]);
    cudaMemcpy(c, d_c, mem_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop[0],s[0]);
    cudaEventSynchronize(stop[0]);

    cudaEventElapsedTime(&time, start[1], stop[1]);
    cudaEventElapsedTime(&time_with_memcpy, start[0], stop[0]);
    cudaEventDestroy(start[0]);
    cudaEventDestroy(stop[0]);
    cudaEventDestroy(start[1]);
    cudaEventDestroy(stop[1]);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}

void stencil_cpu(int* in, int* out, int size, int radius){
    for (int i = 0; i < size; i++){
        int result = 0;
        for (int j = -radius; j <= radius; j++){
            if( ((j<0) && (i < radius)) || ((j>0) && (i > (size - radius)) )){
                result += 0;
            }
            else{
                result += in[i + j];
            }
        }
        out[i] = result;
    }
}

void stencil_cuda(int* in, int* out, int size, int radius, int TB_size, float& time, float& time_with_memcpy){
    int *d_in, *d_out;
    int mem_size = size*sizeof(int);
    cudaStream_t s[2];
    cudaEvent_t start[2], stop[2];
    //float time, time_with_memcpy;

    cudaEventCreate(&start[0]);
    cudaEventCreate(&stop[0]);
    cudaEventCreate(&start[1]);
    cudaEventCreate(&stop[1]);
    cudaStreamCreate(&s[0]);
    cudaStreamCreate(&s[1]);

    cudaMalloc((void **)&d_in, mem_size);
    cudaMalloc((void **)&d_out, mem_size);

    cudaEventRecord(start[0],s[0]);
    cudaMemcpy(d_in, in, mem_size, cudaMemcpyHostToDevice);

    cudaEventRecord(start[1],s[1]);
    int M = TB_size;
    int N = (size + TB_size -1)/TB_size;
    stencil_gpu<<<N,M>>>(d_in, d_out, size, radius);

    cudaEventRecord(stop[1],s[1]);
    cudaEventSynchronize(stop[1]);
    cudaMemcpy(out, d_out, mem_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop[0],s[0]);
    cudaEventSynchronize(stop[0]);

    cudaEventElapsedTime(&time, start[1], stop[1]);
    cudaEventElapsedTime(&time_with_memcpy, start[0], stop[0]);
    cudaEventDestroy(start[0]);
    cudaEventDestroy(stop[0]);
    cudaEventDestroy(start[1]);
    cudaEventDestroy(stop[1]);
    cudaFree(d_in);
    cudaFree(d_out);
}

void stencil_cuda_shared(int* in, int* out, int size, int radius, int TB_size, float& time, float& time_with_memcpy){
    int *d_in, *d_out;
    int mem_size = size*sizeof(int);
    cudaStream_t s[2];
    cudaEvent_t start[2], stop[2];
    //float time, time_with_memcpy;

    cudaEventCreate(&start[0]);
    cudaEventCreate(&stop[0]);
    cudaEventCreate(&start[1]);
    cudaEventCreate(&stop[1]);
    cudaStreamCreate(&s[0]);
    cudaStreamCreate(&s[1]);

    cudaMalloc((void **)&d_in, mem_size);
    cudaMalloc((void **)&d_out, mem_size);

    cudaEventRecord(start[0],s[0]);
    cudaMemcpy(d_in, in, mem_size, cudaMemcpyHostToDevice);

    cudaEventRecord(start[1],s[1]);
    int M = TB_size;
    int N = (size + TB_size -1)/TB_size;
    if(radius==2){
        stencil_shared_r2<<<N,M>>>(d_in, d_out, TB_size);
    }
    else if(radius==4){
        stencil_shared_r4<<<N,M>>>(d_in, d_out, TB_size);
    }
    else if(radius==8){
        stencil_shared_r8<<<N,M>>>(d_in, d_out, TB_size);
    }
    else if(radius==16){
        stencil_shared_r16<<<N,M>>>(d_in, d_out, TB_size);
    }
    //stencil_shared_mem<<<N,M>>>(d_in, d_out, TB_size, radius);

    cudaEventRecord(stop[1],s[1]);
    cudaEventSynchronize(stop[1]);
    cudaMemcpy(out, d_out, mem_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop[0],s[0]);
    cudaEventSynchronize(stop[0]);

    cudaEventElapsedTime(&time, start[1], stop[1]);
    cudaEventElapsedTime(&time_with_memcpy, start[0], stop[0]);
    cudaEventDestroy(start[0]);
    cudaEventDestroy(stop[0]);
    cudaEventDestroy(start[1]);
    cudaEventDestroy(stop[1]);
    cudaFree(d_in);
    cudaFree(d_out);
}


void print_array(int* array, int size){
    for(int i = 0; i < size; i++){
        std::cout << array[i] << ' ';
    }
    std::cout << std::endl;
}
*/
void print_matrix(int* matrix, int size){
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            std::cout << matrix[i*size+j] << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
