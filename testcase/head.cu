#include "head.h"

void printTensor(dt *A,int a,int b,int c){
	for(int i = 0;i<c;i++){
		for(int j = 0;j<a;j++){
			for(int k =0;k<b;k++){
				cout<<A[i*a*b+j*b+k]<<"  ";
			}
			cout<<endl;
		}
		cout<<"-----------------------------------"<<endl;
	}
	cout<<endl;
}

void tranpro(dt *A,dt *B,dt *res,int a,int b,int c){

	dt sum = 0.0;
	dt *At = new dt[a*b]();
	for(int i = 0;i<a;i++){
		for(int j = 0;j<b;j++){
			At[j*a+i] = A[i*b+j];
		}
	} //At is b*a B is a*c
//	printTensor(At,b,a,1);
	for(int i = 0;i<b;i++){
		for(int j = 0;j<c;j++){
			sum = 0.0;
			for(int k = 0;k<a;k++){
				sum+=At[i*a+k]*B[k*c+j];
			}	
			res[i*c+j] = sum;
		}
	}

//	printTensor(res,b,c,1);
	
	delete[] At;At=nullptr;
}

void notranpro(dt *A,dt *B,dt *res,int a,int b,int c){
//A a*b B a*c res b*c
	dt sum = 0.0;

	for(int i = 0;i<b;i++){
		for(int j = 0;j<c;j++){
			sum = 0.0;
			for(int k = 0;k<a;k++){
				sum+=A[k*b+i]*B[k*c+j];
			}	
			res[i*c+j] = sum;
		}
	}

//	printTensor(res,b,c,1);
	
}

void cunoTran(dt *A,dt *B,dt *res,int a,int b,int c){
	dt *d_A = NULL;	
	dt *d_B = NULL;	
	dt *d_res = NULL;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*b); 	//a*b	
	cudaMalloc((void**)&d_B,sizeof(dt)*a*c);	//a*c	
	cudaMalloc((void**)&d_res,sizeof(dt)*c*b);	//b*c
	cudaMemcpy(d_A,A,sizeof(dt)*a*b,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,sizeof(dt)*a*c,cudaMemcpyHostToDevice);
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(
		handle,
		CUBLAS_OP_N,
		CUBLAS_OP_T,
		c,
		b,
		a,
		&alpha,
		d_B,
		c,
		d_A,
		b,
		&beta,
		d_res,
		c
	           );

	cudaMemcpy(res,d_res,sizeof(dt)*c*b,cudaMemcpyDeviceToHost);
	cublasDestroy(handle);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_res);
}
void cuTran(dt *A,dt *B,dt *res,int a,int b,int c){
	dt *d_A = NULL;	
	dt *d_AT = NULL;	
	dt *d_B = NULL;	
	dt *d_res = NULL;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*b); 	//a*b	
	cudaMalloc((void**)&d_AT,sizeof(dt)*a*b); 	//a*b	
	cudaMalloc((void**)&d_B,sizeof(dt)*a*c);	//a*c	
	cudaMalloc((void**)&d_res,sizeof(dt)*c*b);	//b*c
	cudaMemcpy(d_A,A,sizeof(dt)*a*b,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,sizeof(dt)*a*c,cudaMemcpyHostToDevice);
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
		d_A,
		b,
		&beta,
		d_AT,
		a,
		d_AT,
		a
		 );

	cublasSgemm(
		handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		c,
		b,
		a,
		&alpha,
		d_B,
		c,
		d_AT,
		a,
		&beta,
		d_res,
		c
	           );

	cudaMemcpy(res,d_res,sizeof(dt)*c*b,cudaMemcpyDeviceToHost);
	cublasDestroy(handle);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_res);
	cudaFree(d_AT);

}

void cuStrideMode(dt *A,dt *B,dt *res,int a,int b,int c){
	// A is a*b*c B is b*c res is a*c*c
	dt *d_A = NULL;	
	dt *d_B = NULL;	
	dt *d_res = NULL;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*b*c); 	//a*b*c	
	cudaMalloc((void**)&d_B,sizeof(dt)*b*c);	//b*c	
	cudaMalloc((void**)&d_res,sizeof(dt)*a*c*c);	//a*c*c
	cudaMemcpy(d_A,A,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,sizeof(dt)*b*c,cudaMemcpyHostToDevice);
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	clock_t time1,time2;
	time1 = clock();

	cublasSgemmStridedBatched(
		handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		a,
		c,
		b,
		&alpha,
		d_A,
		a,
		a*b,
		d_B,
		b,
		0,
		&beta,
		d_res,
		a,
		a*c,
		c
		);

	time2 = clock();
	cout<<(double)(time2-time1)/CLOCKS_PER_SEC<<"s"<<endl;
	cudaMemcpy(res,d_res,sizeof(dt)*a*c*c,cudaMemcpyDeviceToHost);
	printTensor(res,a,c,c);
	cublasDestroy(handle);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_res);
	
}

void cuStride(dt *A,dt *B,dt *res,int a,int b,int c){
	// A is a*b*c B is b*c res is a*c*c
	dt *d_A = NULL;	
	dt *d_B = NULL;	
	dt *d_res = NULL;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*b*c); 	//a*b*c	
	cudaMalloc((void**)&d_B,sizeof(dt)*b*c*c);	//b*c	
	cudaMalloc((void**)&d_res,sizeof(dt)*a*c*c);	//a*c*c
	cudaMemcpy(d_A,A,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,sizeof(dt)*b*c*c,cudaMemcpyHostToDevice);
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	clock_t time1,time2;
	time1 = clock();

	cublasSgemmStridedBatched(
		handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		a,
		c,
		b,
		&alpha,
		d_A,
		a,
		a*b,
		d_B,
		b,
		b*c,
		&beta,
		d_res,
		a,
		a*c,
	        c
		);

	time2 = clock();
	cout<<(double)(time2-time1)/CLOCKS_PER_SEC<<"s"<<endl;
	cudaMemcpy(res,d_res,sizeof(dt)*a*c*c,cudaMemcpyDeviceToHost);
	cublasDestroy(handle);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_res);

}
