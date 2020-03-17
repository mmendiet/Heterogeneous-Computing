__kernel void directConv(__global int* I_gpu, __global int* K_gpu, __global int* O_gpu, int Ho, int Wo, int k, int nK) {
    int row = get_global_id(0);//*get_global_size(0);
    int col = get_global_id(1);//*get_local_size(1);
    
    int accu = 0;
    if(row<Ho && col<Wo) {
      for (int f=0; f<nK; f++) {
        for(int c=0; c<k; c++) {
          for(int r=0; r<k; r++) {
              accu = accu + I_gpu[(row+c)*(Wo+k-1)+(col+r)] * K_gpu[(c+f)*k+r];
          }
        }
      }
      O_gpu[row*Wo+col] = accu;
    }
  }