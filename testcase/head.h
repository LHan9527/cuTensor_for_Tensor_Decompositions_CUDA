#ifndef GUARD_head_h
#define GUARD_head_h

#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <fstream>

typedef float dt;
using namespace std;

void allin(dt *X,dt *M,dt *N,dt *res,int m,int n,int r,int k,int flag);
__global__ void krpro(dt *M,dt *N,dt *res,int m,int n,int r);
__global__ void elepro(dt *A,dt *B,int r);
void printTensor(dt *A,int a,int b,int c);
void solve(dt *left,dt *right,dt *res,int r,int m);
void cp_als(dt *X,dt *A,dt *B,dt *C,int a,int b,int c,int r);
void KRao(dt *X,dt *M,dt *N,dt *left,dt *right,int m,int n,int r,int k,int flag);
void Btensor2mat(dt *A,dt *A1,dt *A2,dt *A3,int a,int b,int c);
void getvector(dt *A,dt *U,int m,int n,int r);
void getvector1(dt *A,dt *U,int m,int n,int r);
void getvector(dt *A,dt *U,int m,int n,int r);
__global__ void transpose(dt *A,dt *AT,int m,int n);
void tranpro(dt *A,dt *B,dt *res,int a,int b,int c);
void notranpro(dt *A,dt *B,dt *res,int a,int b,int c);
void cunoTran(dt *A,dt *B,dt *res,int a,int b,int c);//cu tr max pro 
void cuTran(dt *A,dt *B,dt *res,int a,int b,int c);//cu max pro
void cuStrideMode(dt *A,dt *B,dt *res,int a,int b,int c);
void cuStride(dt *A,dt *B,dt *res,int a,int b,int c);
void cuStrideModetran(dt *A,dt *B,dt *res,int a,int b,int c);
void cuStrideModenotran(dt *A,dt *B,dt *res,int a,int b,int c);
__global__ void mode1tran(dt *AA,dt *A1,int a,int b,int c);
__global__ void mode2tran(dt *AA,dt *A2,int a,int b,int c);
__global__ void mode3tran(dt *AA,dt *A3,int a,int b,int c);
void newtest(dt *A,dt *U1,dt *U2,dt *U3,dt *res1,int a,int b,int c);
void newtest16(dt *A,dt *U1,dt *U2,dt *U3,dt *res1,int a,int b,int c);

void newtest16h(dt *A,dt *U1,dt *U2,dt *U3,dt *res1,int a,int b,int c);
void newtest32(dt *A,dt *U1,dt *U2,dt *U3,dt *res1,int a,int b,int c);
__global__ void f2h(dt *A,half *B,int num);

void Hosvd(dt *A,dt *res,dt *U1,dt *U2,dt *U3,int a,int b,int c);
__global__ void norm(dt *A,dt *B,int m ,int r);

#endif
	
