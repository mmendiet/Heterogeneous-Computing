#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <chrono>

//#define N 1000
//#define M 512
//nvcc testing.cu -o test
//


__global__ void stencil_1d(int *in, int *out) {
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + radius;
    // Read input elements into shared memory
    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS) {
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }
    // Synchronize (ensure all the data is available)
    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
        result += temp[lindex + offset];
    }
    // Store the result
    out[gindex] = result;
}

__global__ void stencil_noShared(int *in, int *out) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    // Synchronize (ensure all the data is available)
    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
        result += in[gindex + offset];
    }
    // Store the result
    out[gindex] = result;
}


void cpuStencil(int *a, int *b, int *c, int n) {
    for(int i=0; i<n; ++i) {
        c[0] = a[0] + b[0];
    }
}

void random_ints(int* x, int size)
{
	int i;
	for (i=0;i<size;i++) {
		x[i]=rand()%10;
	}
}

int main(int argc, char* argv[]) {
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int R = atoi(argv[4])
    int *a, *b, *c;

    // device copies of a, b, c
    int size = N * sizeof(int);

    // Setup input values
    a = (int*)malloc(size); random_ints(a, N);
    b = (int*)malloc(size); random_ints(b, N);
    c = (int*)malloc(size);  
    if (strcmp(argv[3],"gpu")==0) {
        float total_time = 0;
        for(int i=0; i< 1000; ++i) {
            // host copies of a, b, c
            int *d_a, *d_b, *d_c;
            // Allocate space for device copies of a, b, c
            cudaMalloc((void **)&d_a, size);
            cudaMalloc((void **)&d_b, size);
            cudaMalloc((void **)&d_c, size);

            // Copy inputs to device
            cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
            // Launch add() kernel on GPU

            float time;
            cudaEvent_t start, stop;
            
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord( start, 0 );
            add<<<(N + M-1) / M,M>>>(d_a, d_b, d_c, N);
            cudaEventRecord( stop, 0 );
            cudaEventSynchronize( stop );

            cudaEventElapsedTime( &time, start, stop );
            cudaEventDestroy( start );
            cudaEventDestroy( stop );
            total_time += time;
        
            // Copy result back to host
            cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
            cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        }
        float nanosec = (total_time);
        std::cout << "N: " << N << "   M: " << M << "   GPU time: " << nanosec << "ns" << std::endl;
    }

    else {
        auto t1 = std::chrono::high_resolution_clock::now();
        cpuStencil(a,b,c, N);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count();
        std::cout << "N: " << N << "   M: " << M << "   CPU time: " << duration << "ns" << std::endl;
        //time = 100.0;
    }
    //printf("a[0]: %i, b[0]: %i, c[0]: %i\nGPU Time: %f\n", a[0], b[0], c[0], time);
    //printf("N: %i   M: %i   Time: %f\n", N, M, time);
    // Cleanup
    free(a); free(b); free(c);

    return 0;
}
