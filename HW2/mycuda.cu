#include <stdio.h>
#include "mycuda.h"
#include <iostream>


__global__ void my_kernel(){
  printf("Hello!!\n");
}

__global__ void directConv(int* I_gpu, int* K_gpu, int* O_gpu, int Ho, int Wo, int k) {
  // Row i of matrix C
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  // Column j of matrix C
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  int accu = 0;
  if(row<Ho && col<Wo) {
    for(int c=0; c<k; c++) {
        for(int r=0; r<k; r++) {
            accu = accu + I_gpu[(row+c)*(Wo+k-1)+(col+r)] * K_gpu[c*k+r];
        }
    }
    O_gpu[row*Wo+col] = accu;
  }
}

__global__ void matrixMulCol(int* I_gpu, int* K_gpu, int* O_gpu, int Hc, int nK, int K) {
  // Row i of matrix C
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  // Column j of matrix C
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  int accu = 0;
  if(row<Hc && col<nK) {
    for(int k=0; k<K; k++) {
        accu = accu + I_gpu[k*Hc+row] * K_gpu[k*nK+col];
    }
    O_gpu[row*nK+col] = accu;
  }
}

__global__ void matrixMulCol_shared(int* I_gpu, int* K_gpu, int* O_gpu, int Hc, int nK, int K) {
  // Row i of matrix C
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  // Column j of matrix C
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ int filter[];
  // for(int f=0; f<K*nK; f++) {
  //   filter[f] = K_gpu[f];
  // }
  if(col<nK) {
    for(int f=0; f<K; f++) {
      filter[f] = K_gpu[f*nK+col];
    }
  }

  __syncthreads();

  int accu = 0;
  if(row<Hc && col<nK) {
    for(int k=0; k<K; k++) {
        //accu = accu + I_gpu[k*Hc+row] * filter[k*nK+col];
        accu = accu + I_gpu[k*Hc+row] * filter[k];
    }
    O_gpu[row*nK+col] = accu;
  }
}


float dirConv(int* I_cpu, int* K_cpu, int* O_cpu, int I_size, int K_size, int O_size, int Ho, int Wo, int k) {
  int *I_gpu, *K_gpu, *O_gpu;

  cudaMalloc((void **)&I_gpu, I_size);
  cudaMalloc((void **)&K_gpu, K_size);
  cudaMalloc((void **)&O_gpu, O_size);

  //dim3 dimBlock(16, 16);
  //dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
  dim3 dimBlock(16, 16);
  dim3 dimGrid((Ho+dimBlock.x-1)/dimBlock.x, (Wo+dimBlock.y-1)/dimBlock.y);

  // Copy inputs to device
  cudaMemcpy(I_gpu, I_cpu, I_size, cudaMemcpyHostToDevice);
  cudaMemcpy(K_gpu, K_cpu, K_size, cudaMemcpyHostToDevice);

  float time = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  //std::cout << "MC\n";
  cudaEventRecord( start, 0 );
  //std::cout << K << std::endl;
  directConv<<<dimGrid, dimBlock>>>(I_gpu,K_gpu,O_gpu,Ho,Wo,k);
  //matrixMulCol<<<1, 1>>>(I_gpu,K_gpu,O_gpu,Hc,nK,K);
  cudaEventRecord( stop, 0 );

  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  cudaEventDestroy( start );
  cudaEventDestroy( stop );
  //memcopy C_gpu to C_cpu
  cudaMemcpy(O_cpu, O_gpu, O_size, cudaMemcpyDeviceToHost);
  //stop time
  
  cudaFree(I_gpu); cudaFree(K_gpu); cudaFree(O_gpu);
  float microsec = (time)*1000;
  //std::cout << "k: " << k << "\tGPU time: " << microsec << "us" << std::endl;
  cudaDeviceSynchronize();
  return microsec;
}


float matMul(int* I_cpu, int* K_cpu, int* O_cpu, int I_size, int K_size, int O_size, int Hc, int nK, int K) {
  int *I_gpu, *K_gpu, *O_gpu;

  cudaMalloc((void **)&I_gpu, I_size);
  cudaMalloc((void **)&K_gpu, K_size);
  cudaMalloc((void **)&O_gpu, O_size);

  //dim3 dimBlock(16, 16);
  //dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
  dim3 dimBlock(16, 16);
  dim3 dimGrid((Hc+dimBlock.x-1)/dimBlock.x, (nK+dimBlock.y-1)/dimBlock.y);

  // Copy inputs to device
  cudaMemcpy(I_gpu, I_cpu, I_size, cudaMemcpyHostToDevice);
  cudaMemcpy(K_gpu, K_cpu, K_size, cudaMemcpyHostToDevice);

  float time = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  //std::cout << "MC\n";
  cudaEventRecord( start, 0 );
  //std::cout << K << std::endl;
  matrixMulCol<<<dimGrid, dimBlock>>>(I_gpu,K_gpu,O_gpu,Hc,nK,K);
  //matrixMulCol<<<1, 1>>>(I_gpu,K_gpu,O_gpu,Hc,nK,K);
  cudaEventRecord( stop, 0 );

  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  cudaEventDestroy( start );
  cudaEventDestroy( stop );
  //memcopy C_gpu to C_cpu
  cudaMemcpy(O_cpu, O_gpu, O_size, cudaMemcpyDeviceToHost);
  //stop time
  
  cudaFree(I_gpu); cudaFree(K_gpu); cudaFree(O_gpu);
  float microsec = (time)*1000;
  //std::cout << "K: " << K << "\tGPU time: " << microsec << "us" << std::endl;
  cudaDeviceSynchronize();
  return microsec;
}

float matMul_shared(int* I_cpu, int* K_cpu, int* O_cpu, int I_size, int K_size, int O_size, int Hc, int nK, int K) {
  int *I_gpu, *K_gpu, *O_gpu;

  cudaMalloc((void **)&I_gpu, I_size);
  cudaMalloc((void **)&K_gpu, K_size);
  cudaMalloc((void **)&O_gpu, O_size);

  //dim3 dimBlock(16, 16);
  //dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x, (N+dimBlock.y-1)/dimBlock.y);
  dim3 dimBlock(16, 16);
  dim3 dimGrid((Hc+dimBlock.x-1)/dimBlock.x, (nK+dimBlock.y-1)/dimBlock.y);

  // Copy inputs to device
  cudaMemcpy(I_gpu, I_cpu, I_size, cudaMemcpyHostToDevice);
  cudaMemcpy(K_gpu, K_cpu, K_size, cudaMemcpyHostToDevice);

  float time = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  //std::cout << "MC\n";
  cudaEventRecord( start, 0 );
  //std::cout << K << std::endl;
  cudaFuncSetCacheConfig(matrixMulCol_shared, cudaFuncCachePreferShared);
  matrixMulCol_shared<<<dimGrid, dimBlock, K*sizeof(int)>>>(I_gpu,K_gpu,O_gpu,Hc,nK,K);
  //matrixMulCol_shared<<<dimGrid, dimBlock, K*sizeof(int)>>>(I_gpu,K_gpu,O_gpu,Hc,nK,K);

  cudaEventRecord( stop, 0 );

  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  cudaEventDestroy( start );
  cudaEventDestroy( stop );
  //memcopy C_gpu to C_cpu
  cudaMemcpy(O_cpu, O_gpu, O_size, cudaMemcpyDeviceToHost);
  //stop time
  
  cudaFree(I_gpu); cudaFree(K_gpu); cudaFree(O_gpu);
  float microsec = (time)*1000;
  //std::cout << "K: " << K << "\tGPU time: " << microsec << "us" << std::endl;
  cudaDeviceSynchronize();
  return microsec;
}

void my_cuda_func(){
  my_kernel<<<1,1>>>();
  cudaDeviceSynchronize();
}