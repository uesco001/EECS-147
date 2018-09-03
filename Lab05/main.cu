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
    //set standard seed
    srand(217);

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A, *B, *C;
    size_t A_sz,B_sz; //C_sz;
    unsigned VecSize;
   
    dim3 dim_grid, dim_block;

    if (argc == 1) {
        VecSize = 1000;
      } else if (argc == 2) {
      VecSize = atoi(argv[1]);   
      }
      else {
        printf("\nOh no!\nUsage: ./vecAdd <Size>");
        exit(0);
    }

     A_sz = VecSize;
    B_sz = VecSize;

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    size Of vector: %u x %u\n  ", VecSize);

    // llocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
	cudaMallocManaged((void**) &A, VecSize*sizeof(float));
        cudaMallocManaged((void**) &B, VecSize*sizeof(float));
        cudaMallocManaged((void**) &C, VecSize*sizeof(float));
        for (unsigned int i=0; i < A_sz; i++) { A[i] = (rand()%100)/100.00; }
        for (unsigned int i=0; i < B_sz; i++) { B[i] = (rand()%100)/100.00; }
        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer)); 
   
 // Launch kernel  ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    basicVecAdd(A, B, C, VecSize); //In kernel.cu
    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------
    printf("Verifying results..."); fflush(stdout);
     verify(A, B, C, VecSize);

    // Free memory ------------------------------------------------------------
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;

}
