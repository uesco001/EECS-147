
/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void verify(float *A, float *B, , float *C, unsigned int n) {
   
  const float relativeTolerance = 1e-2;

  for(int i = 0; i < n; ++i) {
      float sum1 = A[i]+B[i];
      printf("\t%f/%f",sum1,C[i]);
      float relativeError1 = (sum1 - C[i])/sum1;
      if (relativeError1 > relativeTolerance
        || relativeError1 < -relativeTolerance) {
        printf("\nTEST FAILED\n\n");
	printf("First one");
       
        exit(0);
      }
    
  }


  }

  printf("\nTEST PASSED\n\n");

}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}
