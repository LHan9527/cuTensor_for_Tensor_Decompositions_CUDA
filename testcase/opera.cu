#include "head.h"

void Btensor2mat(dt *A,dt *A1,dt *A2,dt *A3,int a,int b,int c){
	
	dt *d_AA;
	dt *d_A1;
	dt *d_A2;
	dt *d_A3;

	cudaMalloc((void **)&d_AA,sizeof(dt)*a*b*c);
	cudaMalloc((void **)&d_A1,sizeof(dt)*a*b*c);

	cudaMemcpy(d_AA,A,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);

	dim3 threads(512,1,1);
	dim3 blocks(((a*b*c+512-1)/512),1,1);

	mode1tran<<<blocks,threads>>>(d_AA,d_A1,a,b,c);
	cudaMemcpy(A1,d_A1,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost);
	cudaFree(d_A1);

	cudaMalloc((void **)&d_A2,sizeof(dt)*a*b*c);
	mode2tran<<<blocks,threads>>>(d_AA,d_A2,a,b,c);
	cudaMemcpy(A2,d_A2,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost);
	cudaFree(d_A2);

	cudaMalloc((void **)&d_A3,sizeof(dt)*a*b*c);
	mode3tran<<<blocks,threads>>>(d_AA,d_A3,a,b,c);

	cudaMemcpy(A3,d_A3,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost);

	cudaFree(d_AA);
	cudaFree(d_A3);

	cout<<"Btensor2mat is over"<<endl;

/*		for(int k = 0;k<c;k++){
			for(int i = 0;i<a;i++){
				for(int j = 0;j<b;j++){
					t2m[i*b*c+k*b+j] = A[k*a*b+i*b+j];
				}
			}
		}
	
		for(int k = 0;k<c;k++){
			for(int i = 0;i<a;i++){
				for(int j = 0;j<b;j++){
					t2m[j*a*c+k*a+i] = A[k*a*b+i*b+j];
				}
			}
		}

		for(int k = 0;k<c;k++){
			for(int i = 0;i<a;i++){
				for(int  j = 0;j<b;j++){
					 t2m[k*a*b+j*a+i]= A[k*a*b+i*b+j];
				}
			}
		}

*/

}

void getvector1(dt *A,dt *U,int m,int n,int r){
	dt *d_A;
	cudaMalloc((void**)&d_A,sizeof(dt)*m*n);
	cudaMemcpy(d_A,A,sizeof(dt)*m*n,cudaMemcpyHostToDevice);
	dt *d_C;
	cudaMalloc((void**)&d_C,sizeof(dt)*m*m);
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	cublasSsyrk(
		handle,
		uplo,
		CUBLAS_OP_T,
		m,n,
		&alpha,
		d_A,n,
		&beta,
		d_C,m	
		);

	cusolverDnHandle_t cusolverH = NULL;
	dt *d_V;
	cudaMalloc((void**)&d_V,sizeof(dt)*m*r);
	dt *d_W;
	int *devInfo = NULL;
	dt *d_work = NULL;
	int lwork;
	int info_gpu = 0;
	cusolverDnCreate(&cusolverH);
	cudaMalloc((void**)&devInfo,sizeof(int));
	cudaMalloc((void**)&d_W,sizeof(dt)*m);
	
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	cusolverDnSsyevd_bufferSize(
			cusolverH,
			jobz,
			uplo,
			m,
			d_C,
			m,
			d_W,
			&lwork
			);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);

	cusolverDnSsyevd(
			cusolverH,
			jobz,
			uplo,
			m,
			d_C,   //store vectors
			m,
			d_W,
			d_work,
			lwork,
			devInfo
			);
	cudaDeviceSynchronize();
/*	dt *hh = new dt[m*r]();
	cudaMemcpy(hh,d_C+m*(m-r),sizeof(dt)*m*r,cudaMemcpyDeviceToHost);
	printTensor(hh,m,r,1);
	delete[] hh,hh=nullptr;
	
	dt *Ctemp;
	cudaMalloc((void**)&Ctemp,sizeof(dt)*m*r);
	Ctemp = d_C+m*(m-r);
	*/
//	cudaMemcpy(V,d_C,sizeof(dt)*m*m,cudaMemcpyDeviceToHost);
	cudaMemcpy(&info_gpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
	if(info_gpu == 0){
		cout<<"ok"<<endl;
	}else{
		cout<<info_gpu<<endl;
	}
	// now V is vectors 
	cudaFree(d_W);
	cudaFree(d_A);
	cudaFree(d_work);
	cudaFree(devInfo);
	cusolverDnDestroy(cusolverH);
	
	cublasSgeam(
		handle,
		CUBLAS_OP_T,
		CUBLAS_OP_N,
		r,m,
		&alpha,
		d_C+m*(m-r),
//		Ctemp,
		m,
		&beta,
		d_V,
		r,
		d_V,r
		);
	cudaMemcpy(U,d_V,sizeof(dt)*m*r,cudaMemcpyDeviceToHost);
//	printTensor(U,m,r,1);

	cudaFree(d_C);
	//printTensor(W,m,1,1);
	cudaFree(d_V);
	cublasDestroy(handle);
	cudaDeviceReset();

}

void getvector(dt *A,dt *U,int m,int n,int r){
	//we compute A*A'
	dt *d_A;
	dt *d_AT;
	cudaMalloc((void**)&d_A,sizeof(dt)*m*n);
	cudaMalloc((void**)&d_AT,sizeof(dt)*m*n);
	dt *d_AAT;
	cudaMalloc((void**)&d_AAT,sizeof(dt)*m*m);
	dt alpha = 1.0;
	dt beta = 0.0;
	cudaMemcpy(d_A,A,sizeof(dt)*m*n,cudaMemcpyHostToDevice);
	dim3 threads(512,1,1);
	dim3 blocks((m*n+512-1)/512,1,1);
	transpose<<<blocks,threads>>>(d_A,d_AT,m,n);  // now d_AT n*m
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			m,
			m,
			n,
			&alpha,
			d_AT,
			m,
			d_A,
			n,
			&beta,
			d_AAT,  //store A*A'
			m
			);
	cublasDestroy(handle);
	cudaFree(d_A);
	cudaFree(d_AT);
// eig
	cusolverDnHandle_t cusolverH = NULL;
	dt *V = new dt[m*m]();
	dt *V1 = new dt[r*m]();
	dt *d_W;
	int *devInfo = NULL;
	dt *d_work = NULL;
	int lwork;
	int info_gpu = 0;
	cusolverDnCreate(&cusolverH);
	cudaMalloc((void**)&devInfo,sizeof(int));
	cudaMalloc((void**)&d_W,sizeof(dt)*m);
	
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	cusolverDnSsyevd_bufferSize(
			cusolverH,
			jobz,
			uplo,
			m,
			d_AAT,
			m,
			d_W,
			&lwork
			);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);

	cusolverDnSsyevd(
			cusolverH,
			jobz,
			uplo,
			m,
			d_AAT,   //store vectors
			m,
			d_W,
			d_work,
			lwork,
			devInfo
			);
	cudaDeviceSynchronize();
	cudaMemcpy(V,d_AAT,sizeof(dt)*m*m,cudaMemcpyDeviceToHost);
//	cudaMemcpy(W,d_W,sizeof(dt)*m,cudaMemcpyDeviceToHost);
	cudaMemcpy(&info_gpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
	if(info_gpu == 0){
		cout<<"ok"<<endl;
	}else{
		cout<<info_gpu<<endl;
	}

	cudaFree(d_W);
	cudaFree(d_work);
	cudaFree(devInfo);
	cudaFree(d_AAT);
//	printTensor(V,m,m,1);
//	printTensor(W,m,1,1);
	cusolverDnDestroy(cusolverH);
	cudaDeviceReset();
//	printTensor(V,m,m,1);
	for(int i=0;i<r;i++){
		for(int j = 0;j<m;j++){
			V1[i*m+j] = V[i*m+j+m*(m-r)];
			U[j*r+i] = V1[i*m+j];
		}
	}
//	printTensor(U,m,r,1);
		
	delete[] V;V=nullptr;
	delete[] V1;V1=nullptr;

}






