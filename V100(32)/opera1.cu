#include "opera.h"

void maxpro(dt *A,dt *B,dt *C,int a,int b,int c){
	// A a*b, B b*c  C a*c
	dt *d_A;
	dt *d_B;
	dt *d_C;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*b);
	cudaMalloc((void**)&d_B,sizeof(dt)*b*c);
	cudaMalloc((void**)&d_C,sizeof(dt)*a*c);

	cudaMemcpy(d_A,A,sizeof(dt)*a*b,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,sizeof(dt)*b*c,cudaMemcpyHostToDevice);
	
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasStatus_t cublasStat = cublasCreate(&handle);
	cublasSgemm(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			c,
			a,
			b,
			&alpha,
			d_B,
			c,
			d_A,
			b,
			&beta,
			d_C,  //store A*A'
			c
			);

	cudaMemcpy(C,d_C,sizeof(dt)*a*c,cudaMemcpyDeviceToHost);
	cublasDestroy(handle);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

void V100maxpro(dt *A,dt *B,dt *C,int a,int b,int c){
	cublasHandle_t handle;
	cublasStatus_t cublasStat = cublasCreate(&handle);
	cublasStat = cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
	
	dt *d_A;
	dt *d_B;
	dt *d_C;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*b);
	cudaMalloc((void**)&d_B,sizeof(dt)*b*c);
	cudaMalloc((void**)&d_C,sizeof(dt)*a*c);

	cudaMemcpy(d_A,A,sizeof(dt)*a*b,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,sizeof(dt)*b*c,cudaMemcpyHostToDevice);
//	cublasGemmAlgo_t algo;
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasGemmEx(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			c,
			a,
			b,
			&alpha,
			d_B,
			CUDA_R_16F,
			c,
			d_A,
			CUDA_R_16F,
			b,
			&beta,
			d_C,
			CUDA_R_16F,
			c,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT
			);

	cudaMemcpy(C,d_C,sizeof(dt)*a*c,cudaMemcpyDeviceToHost);
	cublasDestroy(handle);
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
 
void v100mpStride(dt *A,dt *B,dt *C,int a,int b,int c,int r){
	//  A a*b*c   B b*r  C a*r*b
	cublasHandle_t handle;
	cublasStatus_t cublasStat = cublasCreate(&handle);
	cublasStat = cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
	
	dt *d_A;
	dt *d_B;
	dt *d_C;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*b*c);
	cudaMalloc((void**)&d_B,sizeof(dt)*b*r);
	cudaMalloc((void**)&d_C,sizeof(dt)*a*r*c);

	cudaMemcpy(d_A,A,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,sizeof(dt)*b*r,cudaMemcpyHostToDevice);
	
	dt *temp = new dt[a*b*c]();
	dt *temp1 = new dt[a*b*c]();
	dt *d_temp;
	dt *d_temp1;
	cudaMalloc((void**)&d_temp,sizeof(dt)*a*b*c);
	cudaMalloc((void**)&d_temp1,sizeof(dt)*a*b*c);
	dim3 threads(512,1,1);
	dim3 blocks((a*b*c+512-1)/512,1,1);

	mode3tran<<<blocks,threads>>>(d_A,d_temp,a,b,c);
//	cudaMemcpy(temp,d_temp,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost);

	tran3mode<<<blocks,threads>>>(d_temp,d_temp1,a,b,c);
	cudaMemcpy(temp1,d_temp1,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost);

//	printTensor(temp,c,a*b,1);
	printTensor(temp1,a,b,c);
	cudaFree(d_temp);
	cudaFree(d_temp1);
	delete[] temp;temp = nullptr;
	

	dt beta = 1.0;
	dt alpha = 0.0;

	cublasGemmStridedBatchedEx(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			r,
			a,
			b,
			&alpha,
			d_B,
			CUDA_R_16F,
			r,
			0,
			d_A,
			CUDA_R_16F,
			b,
			b*a,
			&beta,
			d_C,
			CUDA_R_16F,
			r,
			r*a,
			c,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT

			);

	cudaMemcpy(C,d_C,sizeof(dt)*a*r*c,cudaMemcpyDeviceToHost);
	cublasDestroy(handle);
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

}
