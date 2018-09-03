/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.cu"

int main (int argc, char *argv[])
{
    srand(217);
    Timer timer;
    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);
    float *A_h, *B_h, *C_h;
    float *A0_d, *B0_d, *C0_d, *A1_d, *B1_d, *C1_d;
    size_t A_sz, B_sz, C_sz;
    unsigned VecSize;
    int SegSize;
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    dim3 dim_grid, dim_block;

    if (argc == 1) {
        VecSize = 1000000;

      } else if (argc == 2) {
      VecSize = atoi(argv[1]);   
      
      
      }
  
      else {
        printf("\nOh no!\nUsage: ./vecAdd <Size>");
        exit(0);
    }
SegSize = VecSize/2;
    A_sz = VecSize;
    B_sz = VecSize;
    C_sz = VecSize;
    A_h = (float*) malloc( sizeof(float)*A_sz );
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    B_h = (float*) malloc( sizeof(float)*B_sz );
    for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    C_h = (float*) malloc( sizeof(float)*C_sz );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    size Of vector: %u x %u\n  ", VecSize);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

cudaHostAlloc((void**) A_h, VecSize*sizeof(float), cudaHostAllocDefault);
cudaHostAlloc((void**) B_h, VecSize*sizeof(float), cudaHostAllocDefault);
cudaHostAlloc((void**) C_h, VecSize*sizeof(float), cudaHostAllocDefault);

cudaMalloc((void**) &A0_d, VecSize*sizeof(float));
cudaMalloc((void**) &B0_d, VecSize*sizeof(float));
cudaMalloc((void**) &C0_d, VecSize*sizeof(float));

cudaMalloc((void**) &A1_d, VecSize*sizeof(float));
cudaMalloc((void**) &B1_d, VecSize*sizeof(float));
cudaMalloc((void**) &C1_d, VecSize*sizeof(float));

cudaStreamSynchronize(stream0);
cudaStreamSynchronize(stream1);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
for(int i = 0; i < VecSize; i+= SegSize*2)
{
	cudaMemcpyAsync(A0_d, A_h + i, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
	cudaMemcpyAsync(B0_d, B_h + i, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
	cudaMemcpyAsync(A1_d, A_h + i + SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream1);	
	cudaMemcpyAsync(B1_d, B_h + i + SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream1);
	
	VecAdd<<<SegSize/256 + 1, 256, 0, stream0>>>(SegSize, A0_d, B0_d, C0_d);
	VecAdd<<<SegSize/256 + 1, 256, 0, stream1>>>(SegSize, A1_d, B1_d, C1_d);	    
	
	cudaMemcpyAsync(C_h + i, C0_d, SegSize*sizeof(float), cudaMemcpyDeviceToHost, stream0);
	cudaMemcpyAsync(C_h + i + SegSize, C1_d, SegSize*sizeof(float), cudaMemcpyDeviceToHost, stream1);

}



    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, VecSize);


    // Free memory ------------------------------------------------------------

cudaFreeHost(A_h);
cudaFreeHost(B_h);
cudaFreeHost(C_h);
cudaFree(A0_d);
cudaFree(B0_d);
cudaFree(C0_d);
cudaFree(A1_d);
cudaFree(B1_d);
cudaFree(C1_d);

    return 0;

}
