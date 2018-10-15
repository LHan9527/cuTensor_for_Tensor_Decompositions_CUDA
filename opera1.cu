#include "opera.h"
void cuinv(dt *A,dt *B,int m){
	dt* d_A;
	dt* d_B;
	cudaMalloc((void**)&d_B,sizeof(dt)*m*m);
	cudaMalloc((void**)&d_A,sizeof(dt)*m*m);
	cudaMemcpy(d_A,A,sizeof(dt)*m*m,cudaMemcpyHostToDevice);
	dt *d_U;
	dt *d_S;
	dt *d_V;
	dt *U = new dt[m*m]();
	dt *S = new dt[m]();
	dt *V = new dt[m*m]();
	int *d_info = NULL;
	int lwork = 0;
	dt *d_work = NULL;
	int info = 0;
	const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	const int econ = 0;

	cusolverDnHandle_t cusolverH = NULL;
	gesvdjInfo_t gesvdj_params = NULL;

	cusolverDnCreate(&cusolverH);
	cusolverDnCreateGesvdjInfo(&gesvdj_params);

	cudaMalloc((void**)&d_U,sizeof(dt)*m*m);	
	cudaMalloc((void**)&d_S,sizeof(dt)*m);	
	cudaMalloc((void**)&d_V,sizeof(dt)*m*m);	
	cudaMalloc((void**)&d_info,sizeof(int));
	cusolverDnSgesvdj_bufferSize(
			cusolverH,
			jobz,
			econ,
			m,
			m,
			d_A,
			m,
			d_S,
			d_U,
			m,
			d_V,
			m,
			&lwork,
			gesvdj_params
			);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnSgesvdj(
			cusolverH,
			jobz,
			econ,
			m,
			m,
			d_A,
			m,
			d_S,
			d_U,
			m,
			d_V,
			m,
			d_work,
			lwork,
			d_info,
			gesvdj_params

			);
	cudaDeviceSynchronize();
	cudaMemcpy(&info,d_info,sizeof(int),cudaMemcpyDeviceToHost);
	if(0 == info){
		cout<<"ok"<<endl;
	}else if(0>info){
		cout<<-info<<"is wrong"<<endl;
	}else{
		cout<<info<<"do not work"<<endl;
	}
	cudaFree(d_A);

//	cudaMemcpy(U,d_U,sizeof(dt)*m*m,cudaMemcpyDeviceToHost);
//	cudaMemcpy(S,d_S,sizeof(dt)*m,cudaMemcpyDeviceToHost);
//	cudaMemcpy(V,d_V,sizeof(dt)*m*m,cudaMemcpyDeviceToHost);

//	printTensor(U,m,m,1);
//	printTensor(S,m,1,1);
//	printTensor(V,m,m,1);
	
	cudaDeviceSynchronize();
	
	dim3 threads(512,1,1);
	dim3 blocks((m*m+512-1)/512,1,1);
	matvec<<<blocks,threads>>>(d_U,d_S,m);
	cudaFree(d_S);

	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasStatus_t cublasStat = cublasCreate(&handle);
	cublasSgemm(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_T,
			m,
			m,
			m,
			&alpha,
			d_U,
			m,
			d_V,
			m,
			&beta,
			d_B,  //store A*A'
			m
			);
	cudaFree(d_U);
	cudaFree(d_V);
	dt *d_BT;
	cudaMalloc((void**)&d_BT,sizeof(dt)*m*m);
	transpose<<<blocks,threads>>>(d_B,d_BT,m,m);

	cudaMemcpy(B,d_BT,sizeof(dt)*m*m,cudaMemcpyDeviceToHost);
	cublasDestroy(handle);
	cudaFree(d_B);
	cudaFree(d_BT);

	cusolverDnDestroy(cusolverH);
	cusolverDnDestroyGesvdjInfo(gesvdj_params);
	cudaDeviceReset();
	
	delete[] U;U = nullptr;
	delete[] S;S = nullptr;
	delete[] V;V = nullptr;

}
