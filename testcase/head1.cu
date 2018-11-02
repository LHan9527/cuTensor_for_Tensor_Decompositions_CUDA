#include "head.h"

void cuStrideModetran(dt *A,dt *B,dt *res,int a,int b,int c){
	dt *d_A = NULL;	
	dt *d_BT = NULL;	
	dt *d_B = NULL;	
	dt *d_res = NULL;
//	dt *BT = new dt[b*a]();
	cudaMalloc((void**)&d_A,sizeof(dt)*a*b*c); 	//a*b	
	cudaMalloc((void**)&d_BT,sizeof(dt)*b*a); 	//a*b	
	cudaMalloc((void**)&d_B,sizeof(dt)*a*b);	//a*c	
	cudaMalloc((void**)&d_res,sizeof(dt)*a*a*c);	//b*c
	cudaMemcpy(d_A,A,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,sizeof(dt)*b*a,cudaMemcpyHostToDevice);
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgeam(
		handle,
		CUBLAS_OP_T,
		CUBLAS_OP_N,
		a,
		b,
		&alpha,
		d_B,
		b,
		&beta,
		d_BT,
		a,
		d_BT,
		a
		 ); //d_BT is b*a d_A is a*b*c d_res is a*a*c
//	cudaMemcpy(BT,d_BT,sizeof(dt)*b*a,cudaMemcpyDeviceToHost);
//	printTensor(BT,b,a,1);
	cublasSgemmStridedBatched(
		handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		a,
		a,
		b,
		&alpha,
		d_BT,
		a,
		0,
		d_A,
		b,
		b*a,
		&beta,
		d_res,
		a,
		a*a,
		c
	           );

	cudaMemcpy(res,d_res,sizeof(dt)*a*a*c,cudaMemcpyDeviceToHost);
//	printTensor(res,a,a,c);
	cublasDestroy(handle);
//	delete[] BT;BT = nullptr;
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_res);
	cudaFree(d_BT);

}

void cuStrideModenotran(dt *A,dt *B,dt *res,int a,int b,int c){
	// A is a*b*c B is b*c res is a*c*c
	dt *d_A = NULL;	
	dt *d_B = NULL;	
	dt *d_res = NULL;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*b*c); 	//a*b*c	
	cudaMalloc((void**)&d_B,sizeof(dt)*b*a);	//b*c	
	cudaMalloc((void**)&d_res,sizeof(dt)*a*a*c);	//a*c*c
	cudaMemcpy(d_A,A,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,sizeof(dt)*b*a,cudaMemcpyHostToDevice);
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
//	clock_t time1,time2;
//	time1 = clock();
// d_A a*b*c d_B a*b res is a*a*c
	cublasSgemmStridedBatched(
		handle,
		CUBLAS_OP_T,
		CUBLAS_OP_N,
		a,
		a,
		b,
		&alpha,
		d_B,
		b,
		0,
		d_A,
		b,
		b*a,
		&beta,
		d_res,
		a,
		a*a,
		c
		);

//	time2 = clock();
//	cout<<(double)(time2-time1)/CLOCKS_PER_SEC<<"s"<<endl;
	cudaMemcpy(res,d_res,sizeof(dt)*a*a*c,cudaMemcpyDeviceToHost);
//	printTensor(res,a,a,c);
	cublasDestroy(handle);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_res);
}
