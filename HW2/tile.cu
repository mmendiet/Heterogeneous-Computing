#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <chrono>

#define TILE_DIM 64

void gpuMemTransfer(int* A_cpu, int* B_cpu, int* C_cpu, int N, int size, bool memCol);
void gpuNoMemTransfer(int* A_cpu, int* B_cpu, int* C_cpu, int N, int size, bool memCol);

// __global__ void matrixMul(int* A_gpu, int* B_gpu, int* C_gpu, int N) {
//     // Row i of matrix C
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     // Column j of matrix C
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     int accu = 0;
//     if(row<N && col<N) {
//         for(int k=0; k<N; k++) {
//             accu = accu + A_gpu[row*N+k] * B_gpu[k*N+col];
//         }
//         C_gpu[row*N+col] = accu;
//     }
// }

__global__ void matrixMul(int* A_cpu, int* B_cpu, int* C_cpu, int N)
{
    int accu = 0;

    int row = blockIdx.y*TILE_DIM + threadIdx.y;
    int col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ int shd_A[TILE_DIM][TILE_DIM];
    __shared__ int shd_B[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + N - 1)/TILE_DIM; k++) {

         if (k*TILE_DIM + threadIdx.x < N && row < N)
             shd_A[threadIdx.y][threadIdx.x] = A_cpu[row*N + k*TILE_DIM + threadIdx.x];
         else
             shd_A[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_DIM + threadIdx.y < N && col < N)
             shd_B[threadIdx.y][threadIdx.x] = B_cpu[(k*TILE_DIM + threadIdx.y)*N + col];
         else
             shd_B[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int k = 0; k<TILE_DIM; k++)
             accu += shd_A[threadIdx.y][k] * shd_B[k][threadIdx.x];

         __syncthreads();
    }

    if (row < N && col < N)
        C_cpu[((blockIdx.y * blockDim.y + threadIdx.y)*N) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = accu;
}

// __global__ void matrixMulCol(int* A_gpu, int* B_gpu, int* C_gpu, int N) {
//     // Row i of matrix C
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     // Column j of matrix C
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
    
//     int accu = 0;
//     if(row<N && col<N) {
//         for(int k=0; k<N; k++) {
//             accu = accu + A_gpu[k*N+row] * B_gpu[k*N+col];
//         }
//         C_gpu[row*N+col] = accu;
//     }
// }

__global__ void matrixMulCol(int* A_cpu, int* B_cpu, int* C_cpu, int N)
{
    int accu = 0;

    int row = blockIdx.y*TILE_DIM + threadIdx.y;
    int col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ int shd_A[TILE_DIM][TILE_DIM];
    __shared__ int shd_B[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + N - 1)/TILE_DIM; k++) {

         if (k*TILE_DIM + threadIdx.x < N && row < N)
             shd_A[threadIdx.y][threadIdx.x] = A_cpu[(k*TILE_DIM + threadIdx.x)*N + row];
         else
             shd_A[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_DIM + threadIdx.y < N && col < N)
             shd_B[threadIdx.y][threadIdx.x] = B_cpu[(k*TILE_DIM + threadIdx.y)*N + col];
         else
             shd_B[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int k=0; k<TILE_DIM; k++)
             accu += shd_A[threadIdx.y][k] * shd_B[k][threadIdx.x];

         __syncthreads();
    }

    if (row < N && col < N)
        C_cpu[((blockIdx.y * blockDim.y + threadIdx.y)*N) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = accu;
}
    
void random_ints(int* x, int size)
    {
        srand(time(0)); 
        int i;
        for (i=0;i<size;i++) {
            x[i]=rand()%10;
            //std::cout << x[i] << " ";
        }
    }

void matrixMulCPU(int* A_cpu, int* B_cpu, int* C_cpu, int N) {
    for(int row=0; row<N; row++) {
        for(int col=0; col<N; col++){
            C_cpu[row*N+col] = 0;
            for(int elm=0; elm<N; elm++) {
                C_cpu[row*N+col] = C_cpu[row*N+col] + A_cpu[row*N+elm] * B_cpu[elm*N+col];
            }
        }
    }
}


int main(int argc, char* argv[]){
    //int N = 3;
    int N = atoi(argv[1]);
    bool memCol = false;
    if (strcmp(argv[4],"MC")==0) {
        memCol=true;
    }
    int NN = N*N;
    //define A_cpu, B_cpu, C_cpu in the CPU memory
    int *A_cpu, *B_cpu, *C_cpu;

    int size = NN * sizeof(int);

    // Setup input values
    //std::cout << "A: ";
    A_cpu = (int*)malloc(size); random_ints(A_cpu, NN);
    //std::cout << "\nB: ";
    B_cpu = (int*)malloc(size); random_ints(B_cpu, NN);
    C_cpu = (int*)malloc(size);  
    if (strcmp(argv[2],"gpu")==0) {
        if(strcmp(argv[3],"MT")==0) {
            gpuMemTransfer(A_cpu, B_cpu, C_cpu, N, size, memCol);
        }
        else {
            gpuNoMemTransfer(A_cpu, B_cpu, C_cpu, N, size, memCol);
        }

    }
    else {
        auto t1 = std::chrono::high_resolution_clock::now();
        matrixMulCPU(A_cpu, B_cpu, C_cpu, N);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
        std::cout << "N: " << N << "\tCPU time: " << duration << "us" << std::endl;
    }
    //std::cout << "\nC: " << C_cpu[0] << " " << C_cpu[1] << " " <<C_cpu[2] << " " << C_cpu[3] << " " << C_cpu[4] <<" " << C_cpu[7] <<" " << C_cpu[8] <<"\n";
    free(A_cpu); free(B_cpu); free(C_cpu);
    return 0;
}
    

void gpuMemTransfer(int* A_cpu, int* B_cpu, int* C_cpu, int N, int size, bool memCol) {
    //define A_gpu, B_gpu, C_gpu in the GPU memory
    //std::cout << "\nMem Tr\n";
    int *A_gpu, *B_gpu, *C_gpu;

    cudaMalloc((void **)&A_gpu, size);
    cudaMalloc((void **)&B_gpu, size);
    cudaMalloc((void **)&C_gpu, size);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);

    float time = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (memCol==true) {
        //std::cout << "MC\n";
        cudaEventRecord( start, 0 );

        // Copy inputs to device
        cudaMemcpy(A_gpu, A_cpu, size, cudaMemcpyHostToDevice);
        cudaMemcpy(B_gpu, B_cpu, size, cudaMemcpyHostToDevice);

        matrixMulCol<<<dimGrid, dimBlock>>>(A_gpu,B_gpu,C_gpu,N);

        //memcopy C_gpu to C_cpu
        cudaMemcpy(C_cpu, C_gpu, size, cudaMemcpyDeviceToHost);
        //stop time
        cudaEventRecord( stop, 0 );
    }
    else {
        //std::cout << "nmc\n";
        cudaEventRecord( start, 0 );

        // Copy inputs to device
        cudaMemcpy(A_gpu, A_cpu, size, cudaMemcpyHostToDevice);
        cudaMemcpy(B_gpu, B_cpu, size, cudaMemcpyHostToDevice);

        matrixMul<<<dimGrid, dimBlock>>>(A_gpu,B_gpu,C_gpu,N);

        //memcopy C_gpu to C_cpu
        cudaMemcpy(C_cpu, C_gpu, size, cudaMemcpyDeviceToHost);
        //stop time
        cudaEventRecord( stop, 0 );
    }
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    cudaFree(A_gpu); cudaFree(B_gpu); cudaFree(C_gpu);
    float microsec = (time)*1000;
    std::cout << "N: " << N << "\tMT\t" << memCol << "\tGPU time: " << microsec << "us" << std::endl;
}

void gpuNoMemTransfer(int* A_cpu, int* B_cpu, int* C_cpu, int N, int size, bool memCol) {
    //define A_gpu, B_gpu, C_gpu in the GPU memory
    //std::cout << "\nNoMem Tr\n";
    int *A_gpu, *B_gpu, *C_gpu;

    cudaMalloc((void **)&A_gpu, size);
    cudaMalloc((void **)&B_gpu, size);
    cudaMalloc((void **)&C_gpu, size);

    // Copy inputs to device
    cudaMemcpy(A_gpu, A_cpu, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B_cpu, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);

    float time = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    if (memCol==true) {
        //std::cout << "MC\n";
        cudaEventRecord( start, 0 );
        matrixMulCol<<<dimGrid, dimBlock>>>(A_gpu,B_gpu,C_gpu,N);
        cudaEventRecord( stop, 0 );
    }
    else {
        //std::cout << "nmc\n";
        cudaEventRecord( start, 0 );
        matrixMul<<<dimGrid, dimBlock>>>(A_gpu,B_gpu,C_gpu,N);
        cudaEventRecord( stop, 0 );
    }
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    //memcopy C_gpu to C_cpu
    cudaMemcpy(C_cpu, C_gpu, size, cudaMemcpyDeviceToHost);
    cudaFree(A_gpu); cudaFree(B_gpu); cudaFree(C_gpu);
    float microsec = (time)*1000;
    std::cout << "N: " << N << "\tnt\t" << memCol << "\tGPU time: " << microsec << "us" << std::endl;
}
