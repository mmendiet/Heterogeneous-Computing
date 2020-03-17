// __global__ void matrixMulCol(int* I_gpu, int* K_gpu, int* O_gpu, int Hc, int nK, int K) {
//     // Row i of matrix C
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     // Column j of matrix C
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
    
//     int accu = 0;
//     if(row<Hc && col<nK) {
//       for(int k=0; k<K; k++) {
//           accu = accu + I_gpu[k*Hc+row] * K_gpu[k*nK+col];
//       }
//       O_gpu[row*nK+col] = accu;
//     }
//   }

__kernel void matrixMulCol(__global int* I_gpu, __global int* K_gpu, __global int* O_gpu, int Hc, int nK, int K) {
    int row = get_global_id(0);//*get_global_size(0);
    int col = get_global_id(1);//*get_local_size(1);

    int accu = 0;
    if(row<Hc && col<nK) {
      for(int k=0; k<K; k++) {
          accu = accu + I_gpu[k*Hc+row] * K_gpu[k*nK+col];
      }
      O_gpu[row*nK+col] = accu;
    }
}