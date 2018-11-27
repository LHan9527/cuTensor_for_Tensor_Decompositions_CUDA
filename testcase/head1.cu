#include "head.h"

void newtest(dt *A,dt *U1,dt *U2,dt *U3,dt *res1,int a,int b,int c){
	int m = a*0.1;
	int n = b*0.1;
	int k = c*0.1;
	dt *d_A;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*b*c);
	cudaMemcpy(d_A,A,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);
	dt *d_U1;
	cudaMalloc((void**)&d_U1,sizeof(dt)*a*m);
	cudaMemcpy(d_U1,U1,sizeof(dt)*a*m,cudaMemcpyHostToDevice);
	dt *d_temp1;
	cudaMalloc((void**)&d_temp1,sizeof(dt)*m*b*c);
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemmStridedBatched(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_T,
			b,
			m,
			a,
			&alpha,
			d_A,
			b,
			b*a,
			d_U1,
			m,
			0,
			&beta,
			d_temp1,
			b,
			b*m,
			c
			);
	// now d_temp1 is b*m*c row storage
	cudaFree(d_U1);
	cudaFree(d_A);
	dt *d_U2;
	cudaMalloc((void**)&d_U2,sizeof(dt)*b*n);
	cudaMemcpy(d_U2,U2,sizeof(dt)*b*n,cudaMemcpyHostToDevice);
	dt *d_temp2;
	cudaMalloc((void**)&d_temp2,sizeof(dt)*m*n*c);

	cublasSgemmStridedBatched(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			n,m,b,
			&alpha,
			d_U2,
			n,
			0,
			d_temp1,
			b,
			b*m,
			&beta,
			d_temp2,
			n,
			n*m,
			c
			);
	// now d_temp3 is m*n*c row storage
	cudaFree(d_U2);
	cudaFree(d_temp1);
	dt *d_U3;
	cudaMalloc((void**)&d_U3,sizeof(dt)*c*k);
	cudaMemcpy(d_U3,U3,sizeof(dt)*c*k,cudaMemcpyHostToDevice);
	dt *d_res1;
	cudaMalloc((void**)&d_res1,sizeof(dt)*m*n*k);

	cublasSgemmStridedBatched(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			1,n*n,c,
			&alpha,
			d_U3,
			1,
			c,
			d_temp2,
			c,
			0,
			&beta,
			d_res1,
			1,
			m*n,
			k
			);
	// now d_res1 is m*n*k row storage
	cudaMemcpy(res1,d_res1,sizeof(dt)*m*n*k,cudaMemcpyDeviceToHost);
	cublasDestroy(handle);
	cudaFree(d_U3);
	cudaFree(d_res1);

	
}
void newtest32(dt *A,dt *U1,dt *U2,dt *U3,dt *res1,int a,int b,int c){
	int m = a*0.1;
	int n = b*0.1;
	int k = c*0.1;
	dt *d_A;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*b*c);
	cudaMemcpy(d_A,A,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);
	dt *d_U1;
	cudaMalloc((void**)&d_U1,sizeof(dt)*a*m);
	cudaMemcpy(d_U1,U1,sizeof(dt)*a*m,cudaMemcpyHostToDevice);
	dt *d_temp1;
	cudaMalloc((void**)&d_temp1,sizeof(dt)*m*b*c);
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
	cublasGemmStridedBatchedEx(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_T,
			b,
			m,
			a,
			&alpha,
			d_A,CUDA_R_32F,
			b,
			b*a,
			d_U1,CUDA_R_32F,
			m,
			0,
			&beta,
			d_temp1,CUDA_R_32F,
			b,
			b*m,
			c,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT
			);
	// now d_temp1 is b*m*c row storage
	cudaFree(d_U1);
	cudaFree(d_A);
	dt *d_U2;
	cudaMalloc((void**)&d_U2,sizeof(dt)*b*n);
	cudaMemcpy(d_U2,U2,sizeof(dt)*b*n,cudaMemcpyHostToDevice);
	dt *d_temp2;
	cudaMalloc((void**)&d_temp2,sizeof(dt)*m*n*c);

	cublasGemmStridedBatchedEx(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			n,m,b,
			&alpha,
			d_U2,CUDA_R_32F,
			n,
			0,
			d_temp1,CUDA_R_32F,
			b,
			b*m,
			&beta,
			d_temp2,CUDA_R_32F,
			n,
			n*m,
			c,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT
			);
	// now d_temp3 is m*n*c row storage
	cudaFree(d_U2);
	cudaFree(d_temp1);
	dt *d_U3;
	cudaMalloc((void**)&d_U3,sizeof(dt)*c*k);
	cudaMemcpy(d_U3,U3,sizeof(dt)*c*k,cudaMemcpyHostToDevice);
	dt *d_res1;
	cudaMalloc((void**)&d_res1,sizeof(dt)*m*n*k);

	cublasGemmStridedBatchedEx(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			1,n*n,c,
			&alpha,
			d_U3,CUDA_R_32F,
			1,
			c,
			d_temp2,CUDA_R_32F,
			c,
			0,
			&beta,
			d_res1,CUDA_R_32F,
			1,
			m*n,
			k,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT
			);
	// now d_res1 is m*n*k row storage
	cudaMemcpy(res1,d_res1,sizeof(dt)*m*n*k,cudaMemcpyDeviceToHost);
	cublasDestroy(handle);
	cudaFree(d_U3);
	cudaFree(d_res1);

	
}

void newtest16(dt *A,dt *U1,dt *U2,dt *U3,dt *res1,int a,int b,int c){
	int m = a*0.1;
	int n = b*0.1;
	int k = c*0.1;
	dt *d_A;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*b*c);
	cudaMemcpy(d_A,A,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);
	dt *d_U1;
	cudaMalloc((void**)&d_U1,sizeof(dt)*a*m);
	cudaMemcpy(d_U1,U1,sizeof(dt)*a*m,cudaMemcpyHostToDevice);
	dt *d_temp1;
	cudaMalloc((void**)&d_temp1,sizeof(dt)*m*b*c);
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
	half *h_A;
	cudaMalloc((void**)&h_A,sizeof(half)*a*b*c);
	half *h_U1;
	cudaMalloc((void**)&h_U1,sizeof(half)*a*m);
	dim3 threads(512,1,1);
	dim3 block1((a*b*c/512+512-1),1,1);
	f2h<<<block1,threads>>>(d_A,h_A,a*b*c);
	dim3 block2((a*m/512+512-1),1,1);
	f2h<<<block2,threads>>>(d_U1,h_U1,a*m);
	cudaFree(d_U1);
	cudaFree(d_A);

	cublasGemmStridedBatchedEx(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_T,
			b,
			m,
			a,
			&alpha,
			h_A,
			CUDA_R_16F,
			b,
			b*a,
			h_U1,
			CUDA_R_16F,
			m,
			0,
			&beta,
			d_temp1,
			CUDA_R_32F,
			b,
			b*m,
			c,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT
			);
	// now d_temp1 is b*m*c row storage
	cudaFree(h_U1);
	cudaFree(h_A);
	dt *d_U2;
	cudaMalloc((void**)&d_U2,sizeof(dt)*b*n);
	cudaMemcpy(d_U2,U2,sizeof(dt)*b*n,cudaMemcpyHostToDevice);
	half *h_U2;
	cudaMalloc((void**)&h_U2,sizeof(half)*b*n);
	half *h_temp1;
	cudaMalloc((void**)&h_temp1,sizeof(half)*b*m*c);
	dim3 block3((m*b*c/512+512-1),1,1);
	f2h<<<block3,threads>>>(d_temp1,h_temp1,m*b*c);
	dim3 block4((b*n/512+512-1),1,1);
	f2h<<<block4,threads>>>(d_U2,h_U2,b*n);
	cudaFree(d_U2);
	cudaFree(d_temp1);
	dt *d_temp2;
	cudaMalloc((void**)&d_temp2,sizeof(dt)*m*n*c);

	cublasGemmStridedBatchedEx(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			n,m,b,
			&alpha,
			h_U2,
			CUDA_R_16F,
			n,
			0,
			h_temp1,
			CUDA_R_16F,
			b,
			b*m,
			&beta,
			d_temp2,
			CUDA_R_32F,
			n,
			n*m,
			c,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT
			);
	// now d_temp3 is m*n*c row storage
	cudaFree(h_U2);
	cudaFree(h_temp1);
	dt *d_U3;
	cudaMalloc((void**)&d_U3,sizeof(dt)*c*k);
	cudaMemcpy(d_U3,U3,sizeof(dt)*c*k,cudaMemcpyHostToDevice);

	half *h_U3;
	cudaMalloc((void**)&h_U3,sizeof(half)*c*k);
	half *h_temp2;
	cudaMalloc((void**)&h_temp2,sizeof(half)*n*m*c);
	dim3 block5((c*k/512+512-1),1,1);
	f2h<<<block5,threads>>>(d_U3,h_U3,c*k);
	dim3 block6((m*n*c/512+512-1),1,1);
	f2h<<<block6,threads>>>(d_temp2,h_temp2,m*n*c);
	cudaFree(d_U3);
	cudaFree(d_temp2);
	dt *d_res1;
	cudaMalloc((void**)&d_res1,sizeof(dt)*m*n*k);

	cublasGemmStridedBatchedEx(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			1,n*n,c,
			&alpha,
			h_U3,
			CUDA_R_16F,
			1,
			c,
			h_temp2,
			CUDA_R_16F,
			c,
			0,
			&beta,
			d_res1,
			CUDA_R_32F,
			1,
			m*n,
			k,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT
			);
	// now d_res1 is m*n*k row storage
	cudaMemcpy(res1,d_res1,sizeof(dt)*m*n*k,cudaMemcpyDeviceToHost);
	cublasDestroy(handle);
	cudaFree(h_U3);
	cudaFree(h_temp2);
	cudaFree(d_res1);

	
}


void newtest16h(dt *A,dt *U1,dt *U2,dt *U3,dt *res1,int a,int b,int c){
	int m = a*0.1;
	int n = b*0.1;
	int k = c*0.1;
	dt *d_A;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*b*c);
	cudaMemcpy(d_A,A,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);
	dt *d_U1;
	cudaMalloc((void**)&d_U1,sizeof(dt)*a*m);
	cudaMemcpy(d_U1,U1,sizeof(dt)*a*m,cudaMemcpyHostToDevice);
	half *d_temp1;
	cudaMalloc((void**)&d_temp1,sizeof(half)*m*b*c);
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
	half *h_A;
	cudaMalloc((void**)&h_A,sizeof(half)*a*b*c);
	half *h_U1;
	cudaMalloc((void**)&h_U1,sizeof(half)*a*m);
	dim3 threads(512,1,1);
	dim3 block1((a*b*c/512+512-1),1,1);
	f2h<<<block1,threads>>>(d_A,h_A,a*b*c);
	dim3 block2((a*m/512+512-1),1,1);
	f2h<<<block2,threads>>>(d_U1,h_U1,a*m);
	cudaFree(d_U1);
	cudaFree(d_A);

	cublasGemmStridedBatchedEx(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_T,
			b,
			m,
			a,
			&alpha,
			h_A,
			CUDA_R_16F,
			b,
			b*a,
			h_U1,
			CUDA_R_16F,
			m,
			0,
			&beta,
			d_temp1,
			CUDA_R_16F,
			b,
			b*m,
			c,
			CUDA_R_16F,
			CUBLAS_GEMM_DEFAULT
			);
	// now d_temp1 is b*m*c row storage
	cudaFree(h_U1);
	cudaFree(h_A);
	dt *d_U2;
	cudaMalloc((void**)&d_U2,sizeof(dt)*b*n);
	cudaMemcpy(d_U2,U2,sizeof(dt)*b*n,cudaMemcpyHostToDevice);
	half *h_U2;
	cudaMalloc((void**)&h_U2,sizeof(half)*b*n);
	dim3 block4((b*n/512+512-1),1,1);
	f2h<<<block4,threads>>>(d_U2,h_U2,b*n);
	cudaFree(d_U2);
	half *d_temp2;
	cudaMalloc((void**)&d_temp2,sizeof(half)*m*n*c);

	cublasGemmStridedBatchedEx(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			n,m,b,
			&alpha,
			h_U2,
			CUDA_R_16F,
			n,
			0,
			d_temp1,
			CUDA_R_16F,
			b,
			b*m,
			&beta,
			d_temp2,
			CUDA_R_16F,
			n,
			n*m,
			c,
			CUDA_R_16F,
			CUBLAS_GEMM_DEFAULT
			);
	// now d_temp3 is m*n*c row storage
	cudaFree(h_U2);
	cudaFree(d_temp1);
	dt *d_U3;
	cudaMalloc((void**)&d_U3,sizeof(dt)*c*k);
	cudaMemcpy(d_U3,U3,sizeof(dt)*c*k,cudaMemcpyHostToDevice);

	half *h_U3;
	cudaMalloc((void**)&h_U3,sizeof(half)*c*k);
	dim3 block5((c*k/512+512-1),1,1);
	f2h<<<block5,threads>>>(d_U3,h_U3,c*k);
	cudaFree(d_U3);
	dt *d_res1;
	cudaMalloc((void**)&d_res1,sizeof(dt)*m*n*k);

	cublasGemmStridedBatchedEx(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			1,n*n,c,
			&alpha,
			h_U3,
			CUDA_R_16F,
			1,
			c,
			d_temp2,
			CUDA_R_16F,
			c,
			0,
			&beta,
			d_res1,
			CUDA_R_32F,
			1,
			m*n,
			k,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT
			);
	// now d_res1 is m*n*k row storage
	cudaMemcpy(res1,d_res1,sizeof(dt)*m*n*k,cudaMemcpyDeviceToHost);
	cublasDestroy(handle);
	cudaFree(h_U3);
	cudaFree(d_temp2);
	cudaFree(d_res1);

	
}





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
