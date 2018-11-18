#ifndef HEAD_h
#define HEAD_h
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <cublas_v2.h>
#include "Eigen/Dense"

void tucker(float* X,float *G,float *h_A1,float *h_A2,float *h_A3,int I,int J,int K);

#endif
