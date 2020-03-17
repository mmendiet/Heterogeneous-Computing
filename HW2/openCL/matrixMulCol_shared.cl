__kernel void matrixMulCol_shared(__global int* I_gpu, __global int* K_gpu, __global int* O_gpu, int Hc, int nK, int K, __local int* filter) {
    int row = get_global_id(0);//*get_global_size(0);
    int col = get_global_id(1);//*get_local_size(1);

    if(col<nK) {
        filter[col*K] = K_gpu[nK+col];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int accu = 0;
    if(row<Hc && col<nK) {
      for(int k=0; k<K; k++) {
          accu = accu + I_gpu[k*Hc+row] * filter[col*K+k];
      }
      O_gpu[row*nK+col] = accu;
    }
}