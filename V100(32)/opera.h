/*************************************************************************
	> File Name: opera.h
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年09月09日 星期日 20时44分24秒
 ************************************************************************/
#ifndef GUARD_opera_h
#define GUARD_opera_h

#include <iostream>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <time.h>
#include "cuda_fp16.h"


typedef float dt;

using namespace std;

void HOSVD(dt *A,dt *core,dt *U1,dt *U2,dt *U3,int a,int b,int c);

void SBgemm(dt *A,dt *core,dt *U1,dt *U2,dt *U3,int a,int b,int c);
void printTensor(dt *A,int a,int b,int c);

dt* tensor2mat(dt *A,int a,int b,int c,int mode);
void msvd(dt *A,dt *U,int m,int n,int r);
extern __global__ void transpose(dt* A,dt* AT,int m,int n);

void Btensor2mat(dt *A,dt *A1,dt *A2,dt *A3,int a,int b,int c);
extern __global__ void mode1tran(dt* AA,dt* A1,int a,int b,int c);
extern __global__ void mode2tran(dt* AA,dt* A2,int a,int b,int c);
extern __global__ void mode3tran(dt* AA,dt* A3,int a,int b,int c);
extern __global__ void tran3mode(dt *AA,dt *A3,int a,int b,int c);

void Recover(dt *core,dt *rec,dt *U1,dt *U2,dt *U3,int a ,int b,int c);

void getvector(dt *A,dt *U,int m,int n,int r);


void KRao(dt *M,dt *N,dt *res,int m,int n,int r);
extern __global__ void krpro(dt *M,dt *N,dt *res,int m,int n,int r);

void KRao(dt *X,dt *M,dt *N,dt *left,dt *right,int m,int n,int r,int k,int flag);

extern __global__ void elepro(dt *A,dt *B,int r);


void solve(dt *left,dt *right,dt *res,int r,int m);
extern __global__ void norm(dt *A,dt *B,int m ,int r);

void cp_als(dt *X,dt *A,dt *B,dt *C,int a,int b,int c,int r);

void maxpro(dt *A,dt *B,dt *C,int a,int b,int c);
void V100maxpro(dt *A,dt *B,dt *C,int a,int b,int c);
void v100mpStride(dt *A,dt *B,dt *C,int a,int b,int c,int r);

extern __global__ void f2h(dt *A,half *B,int m);
#endif
