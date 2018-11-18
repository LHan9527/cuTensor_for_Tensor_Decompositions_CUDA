#include "head.h"
#include <cublas_v2.h>

void tucker(float* X,float *G,float *h_A1,float *h_A2,float *h_A3,int I,int J,int K){
	int P = 10;
	int Q = 10;
	int R = 10;
	size_t sd = sizof(float);


float *d_X; 
cudaMalloc((void **)&d_X, I*J*K*sd);
cublasSetVector(I*J*K,sd,X,1,d_X,1);

float *d_G; 
cudaMalloc((void **)&d_G,P*Q*R*sd);

float *d_A1, *d_A2, *d_A3;
   cudaMalloc((void **)&d_A1, I*P*sd);
   cudaMalloc((void **)&d_A2, J*Q*sd);
   cudaMalloc((void **)&d_A3, K*R*sd);

   cublasSetVector(I*P,sd,h_A1,1,d_A1,1);
   cublasSetVector(J*Q,sd,h_A2,1,d_A2,1);
   cublasSetVector(K*R,sd,h_A3,1,d_A3,1);

float *d_temp_3;
 cudaMalloc((void **)&d_temp_3,P*J*K*sd);

 float *d_temp_4;
 cudaMalloc((void **)&d_temp_4,P*J*K*sd);

cublasHandle_t handle; cublasCreate(&handle);  
  float alpha = 1.0f, beta = 0.0f;

  cublasSgemmBatched(handle, CUBLAS_OP_T,CUBLAS_OP_N,P,J,I,&alpha,d_A1,I,0,d_X,I,I*J,&beta,d_temp_3,P,P*J,K);
  cublasSgemmBatched(handle, CUBLAS_OP_N,CUBLAS_OP_N,P,J,K,&alpha,d_temp_3,P,P*J,d_A3,K,0,&beta,d_temp_4,P,P*J,R);
  cublasSgemmBatched(handle, CUBLAS_OP_N,CUBLAS_OP_N,P,Q,J,&alpha,d_temp_4,P,P*J,d_A2,J,0,&beta,d_G,P,P*Q,R);

  cublasGetVector(P*Q*R,sd,d_G,1,G,1); 

cudaFree(d_A1);cudaFree(d_A2);cudaFree(d_A3);
cudaFree(d_temp_3);cudaFree(d_temp_4)
  cudaFree(d_G);
cudaFree(d_X);
  cublasDestroy(handle);
}
