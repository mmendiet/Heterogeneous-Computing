#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <chrono>

//#define N 1000
//#define M 512
//nvcc testing.cu -o test
#define BLOCK_SIZE 128
#define Rd 4


__global__ void stencil_1d(int *in, int *out, int RADIUS) {
    __shared__ int temp[BLOCK_SIZE + 2 * Rd];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;
    // Read input elements into shared memory
    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS && blockIdx.x>0) {
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }
    else if(threadIdx.x < RADIUS && blockIdx.x<=0) {
        temp[lindex - RADIUS] = 0;
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++) {
        result += temp[lindex + offset];
    }
    // Store the result
    out[gindex] = result;
}

__global__ void stencil_noMem(int *in, int *out, int RADIUS, int n) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++) {
        int idx = gindex + offset;
        if(idx>0 && idx<n) {
            result += in[idx];
        }
    }
    // Store the result
    out[gindex] = result;
}


void cpuStencil(int *in, int *out, int RADIUS, int n) {
    // Apply the stencil
    for (int i=0; i < n; i++) {
        for (int offset = -RADIUS ; offset <= RADIUS ; offset++) {
            out[i] += in[i + offset];
        }
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
    int R = atoi(argv[2]);
    int *in, *out;

    // device copies of a, b, c
    int size = N * sizeof(int);

    // Setup input values
    in = (int*)malloc(size); random_ints(in, N);
    out = (int*)malloc(size); random_ints(out, N);
    if (strcmp(argv[3],"gpu")==0) {
        // host copies of a, b, c
        int *d_in, *d_out;

        float milli;
        cudaEvent_t start, stop;
        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord( start, 0 );
        // Allocate space for device copies of a, b, c
        cudaMalloc((void **)&d_in, size);
        cudaMalloc((void **)&d_out, size);

        // Copy inputs to device
        cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);
        // Launch add() kernel on GPU

     

        //stencil_1d<<<(N + BLOCK_SIZE-1) / BLOCK_SIZE,BLOCK_SIZE>>>(d_in, d_out, R);

        stencil_noMem<<<(N + BLOCK_SIZE-1) / BLOCK_SIZE,BLOCK_SIZE>>>(d_in, d_out, R, N);
        
        cudaEventRecord( stop, 0 );
        cudaEventSynchronize( stop );

        cudaEventElapsedTime( &milli, start, stop );
        cudaEventDestroy( start );
        cudaEventDestroy( stop );
    
        // Copy result back to host
        cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
        cudaFree(d_in); cudaFree(d_out);
        float nanosec = (milli)*1000000;
        std::cout << "N: " << N << "   R: " << R << "   GPU time: " << nanosec << " ns" << std::endl;
    }

    else {
        auto t1 = std::chrono::high_resolution_clock::now();
        cpuStencil(in,out,R, N);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count();
        std::cout << "N: " << N << "   R: " << R << "   CPU time: " << duration << " ns" << std::endl;
        //time = 100.0;
    }
    //printf("a[0]: %i, b[0]: %i, c[0]: %i\nGPU Time: %f\n", a[0], b[0], c[0], time);
    //printf("N: %i   M: %i   Time: %f\n", N, M, time);
    // Cleanup
    free(in); free(out);

    return 0;
}
