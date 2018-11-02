#ifndef GUARD_head_h
#define GUARD_head_h

#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

typedef float dt;
using namespace std;

void printTensor(dt *A,int a,int b,int c);

void tranpro(dt *A,dt *B,dt *res,int a,int b,int c);
void notranpro(dt *A,dt *B,dt *res,int a,int b,int c);
void cunoTran(dt *A,dt *B,dt *res,int a,int b,int c);//cu tr max pro 
void cuTran(dt *A,dt *B,dt *res,int a,int b,int c);//cu max pro
void cuStrideMode(dt *A,dt *B,dt *res,int a,int b,int c);
void cuStride(dt *A,dt *B,dt *res,int a,int b,int c);
void cuStrideModetran(dt *A,dt *B,dt *res,int a,int b,int c);
void cuStrideModenotran(dt *A,dt *B,dt *res,int a,int b,int c);
#endif
	
