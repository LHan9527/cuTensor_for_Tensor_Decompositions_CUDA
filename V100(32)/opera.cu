
#include <stdlib.h>
#include "opera.h"
void SBgemm(dt *A,dt *core,dt *U1,dt *U2,dt *U3,int a,int b,int c){
	int r1 = a/8;
	int r2 = b/8;
	int r3 = c/8;
	while(r1%8!=0){
		r1--;
	}
	while(r2%8!=0){
		r2--;
	}
	while(r3%8!=0){
		r3--;
	}
cout<<"come in SBgemm"<<endl;
//	int r1 = 2;
//	int r2 = 3;
//	int r3 = 2;
//	dt *temp1 = new dt[b*r1*c]();
//	dt *temp2 = new dt[r1*r2*c]();
//	dt *temp3 = new dt[c*r1*r2]();	//mode3 mat
	//dt *temp = new dt[r3*r1*r2]();	//result to be convert

	//compute A ×1 U1'×2 U2'×3 U3'
	// first compute U1'[X1,X2,X3～～Xc]U2 = temp;   then temp*U3'   
	//  U1 a*r1  U1' r1*a
	// A a*b*c
	// U2 b*r2  U2' r2*b
	// U3 c*r3  U3' r3*c

	dt alpha = 1.0;
	dt beta = 0.0;
	dt *d_A;
	dt *d_U1;
	dt *d_U2;
	dt *d_U3;
	dt *d_temp1;
	dt *d_temp2;
	dt *d_temp3;
	dt *d_temp;
	cudaMalloc((void **)&d_A,a*b*c*sizeof(dt));
	cudaMalloc((void **)&d_U1,a*r1*sizeof(dt));
	cudaMalloc((void **)&d_temp1,sizeof(dt)*b*r1*c);

	cudaMemcpy(d_A,A,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);
	cudaMemcpy(d_U1,U1,sizeof(dt)*a*r1,cudaMemcpyHostToDevice);



	//cudaMemcpy(d_C,C,sizeof(dt)*a*d*c);

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
	
	dim3 threads(512,1,1);
	

	cublasGemmStridedBatchedEx(
			
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_T,
			b,				//row of A C
			r1,				//col of B C
			a,				//col of A ,row of B
			&alpha,
			d_A,
			CUDA_R_32F,
			b,				//leading dimension store A
			b*a,			//step between two mat
		   	d_U1,
			CUDA_R_32F,
			r1,
			0,
			&beta,
			d_temp1,
			CUDA_R_32F,
			b,
			b*r1,
			c,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT

			);
	cudaFree(d_U1);
	cudaFree(d_A);
	//now d_temp1 store the real value col first
//	cudaMemcpy(temp1,d_temp1,sizeof(dt)*b*r1*c,cudaMemcpyDeviceToHost);
//	printTensor(temp1,r1,b,c);

	cudaMalloc((void **)&d_U2,b*r2*sizeof(dt));
	cudaMalloc((void **)&d_temp2,sizeof(dt)*r1*r2*c);
	cudaMemcpy(d_U2,U2,sizeof(dt)*b*r2,cudaMemcpyHostToDevice);

	

	cublasGemmStridedBatchedEx(
			
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			r2,
			r1,
			b,
			&alpha,
			d_U2,
			CUDA_R_32F,
			r2,
			0,
			d_temp1,
			CUDA_R_32F,
			b,
			b*r1,
			&beta,
			d_temp2,
			CUDA_R_32F,
			r2,
			r2*r1,
			c,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT

			);

//	cudaMemcpy(temp2,d_temp2,sizeof(dt)*r1*r2*c,cudaMemcpyDeviceToHost);
	
//	printTensor(temp2,r1,r2,c);
	cudaFree(d_U2);
	cudaFree(d_temp1);

	cudaMalloc((void **)&d_temp3,sizeof(dt)*r1*r2*c);	//mode 3 mat

	// now temp2 store the real value 
	//we will mat3,and the 
	dim3 blocks((r1*r2*c+512-1)/512,1,1);
	mode3tran<<<blocks,threads>>>(d_temp2,d_temp3,r1,r2,c);
//	temp3 = tensor2mat(temp2,r1,r2,c,3);
//	printTensor(temp3,c,r1*r2,1);
	
	cudaFree(d_temp2);

//	cudaMemcpy(d_temp3,temp3,sizeof(dt)*c*r1*r2,cudaMemcpyHostToDevice);

	cudaMalloc((void **)&d_U3,c*r3*sizeof(dt));
	cudaMemcpy(d_U3,U3,sizeof(dt)*c*r3,cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_temp,sizeof(dt)*r1*r2*r3);

	

	cublasGemmEx(
			
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_T,
			r1*r2,
			r3,
			c,
			&alpha,
			d_temp3,
			CUDA_R_32F,
			r1*r2,
			d_U3,
			CUDA_R_32F,
			r3,
			&beta,
			d_temp,
			CUDA_R_32F,
			r1*r2,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT

			);

	cudaFree(d_U3);
	cudaFree(d_temp3);

	dim3 blocks1((r1*r2*r3+512-1)/512,1,1);
	dt *d_core;
	cudaMalloc((void**)&d_core,sizeof(dt)*r1*r2*r3);
	tran3mode<<<blocks1,threads>>>(d_temp,d_core,r1,r2,r3);
	cudaFree(d_temp);
	cudaMemcpy(core,d_core,sizeof(dt)*r3*r1*r2,cudaMemcpyDeviceToHost);

//	printTensor(temp,r3,r1*r2,1);
	cudaFree(d_core);
	cublasDestroy(handle);
	//mode3 to Tensor 
	//r3 * r1×r2   to r1*r2*r3
/*	for(int k = 0;k<r3;k++){
		for(int i = 0;i<r1;i++){
			for(int j = 0;j<r2;j++){
				core[k*r1*r2+i*r2+j] = temp[k*r1*r2+j*r1+i];
			}
		}
	}
*/

//	delete[] temp1; temp1 = nullptr;
//	delete[] temp2; temp2 = nullptr;
//	delete[] temp3; temp3 = nullptr;
//	delete[] temp; temp = nullptr;
	
	cout<<"SBgeem is over"<<endl;
}

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

dt* tensor2mat(dt *A,int a,int b,int c,int mode){
	dt *t2m = new dt[a*b*c]();
	if (mode == 1){
		for(int k = 0;k<c;k++){
			for(int i = 0;i<a;i++){
				for(int j = 0;j<b;j++){
					t2m[i*b*c+k*b+j] = A[k*a*b+i*b+j];
				}
			}
		}
		return t2m;
	}
	
	if (mode == 2){
		for(int k = 0;k<c;k++){
			for(int i = 0;i<a;i++){
				for(int j = 0;j<b;j++){
					t2m[j*a*c+k*a+i] = A[k*a*b+i*b+j];
				}
			}
		}
		return t2m;
	}

	if (mode == 3){
		for(int k = 0;k<c;k++){
			for(int i = 0;i<a;i++){
				for(int  j = 0;j<b;j++){
					 t2m[k*a*b+j*a+i]= A[k*a*b+i*b+j];
				}
			}
		}
		return t2m;
	}

	return t2m;
}

void msvd(dt *A,dt *U,int m,int n,int r){
	//printTensor(A,m,n,1);
	//printTensor(A,m,n,1);
	cout<<"come in svd"<<endl;
	dt* d_A;
	dt* d_AT;
	cudaMalloc((void**)&d_AT,sizeof(dt)*m*n);
	cudaMalloc((void**)&d_A,sizeof(dt)*m*n);
	cudaMemcpy(d_A,A,sizeof(dt)*m*n,cudaMemcpyHostToDevice);
	dim3 threads(512,1,1);
	dim3 blocks(((m*n+512-1)/512),1,1);
	transpose<<<blocks,threads>>>(d_A,d_AT,m,n);	//mow d_A store the transpose of A
	//cudaMemcpy(A,d_AT,sizeof(dt)*m*n,cudaMemcpyDeviceToHost);
	// the we will use SVD  d_A col-store m*n
	dt* d_Utemp;			//left singular vectors
	cudaMalloc((void**)&d_Utemp,sizeof(dt)*r*m);
	//dt* S = new dt[n];
	dt *d_U;
	dt *d_S;
	dt *d_V;
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
	cudaMalloc((void**)&d_S,sizeof(dt)*n);	
	cudaMalloc((void**)&d_V,sizeof(dt)*n*n);	
	cudaMalloc((void**)&d_info,sizeof(int));
	cusolverDnSgesvdj_bufferSize(
			cusolverH,
			jobz,
			econ,
			m,
			n,
			d_AT,
			m,
			d_S,
			d_U,
			m,
			d_V,
			n,
			&lwork,
			gesvdj_params
			);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnSgesvdj(
			cusolverH,
			jobz,
			econ,
			m,
			n,
			d_AT,
			m,
			d_S,
			d_U,
			m,
			d_V,
			n,
			d_work,
			lwork,
			d_info,
			gesvdj_params

			);
	cudaDeviceSynchronize();
	//cudaMemcpy(Utemp,d_U,sizeof(dt)*m*m,cudaMemcpyDeviceToHost);
	cudaMemcpy(&info,d_info,sizeof(int),cudaMemcpyDeviceToHost);
	if(0 == info){
		cout<<"ok"<<endl;
	}else if(0>info){
		cout<<-info<<"is wrong"<<endl;
	}else{
		cout<<info<<"do not work"<<endl;
	}

	dim3 thread1(512,1,1);
	dim3 block1(max(((m*n+512-1)/512),65535),1,1);
	transpose<<<block1,thread1>>>(d_U,d_Utemp,r,m);
	cudaMemcpy(U,d_Utemp,sizeof(dt)*m*r,cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
//	printTensor(U,m,r,1);

	cudaFree(d_A);
	cudaFree(d_AT);
	cudaFree(d_Utemp);
	cudaFree(d_U);
	cudaFree(d_S);
	cudaFree(d_V);
	cudaFree(d_work);
	cudaFree(d_info);
	cusolverDnDestroy(cusolverH);
	cusolverDnDestroyGesvdjInfo(gesvdj_params);
	cudaDeviceReset();

	
}

void HOSVD(dt *A,dt *core,dt *U1,dt *U2,dt *U3,int a,int b,int c){
	int r1 = a/8;
	int r2 = b/8;
	int r3 = c/8;
	while(r1%8!=0){
		r1--;
	}
	while(r2%8!=0){
		r2--;
	}
	while(r3%8!=0){
		r3--;
	}
//	int r1 = 2;
//	int r2 = 3;
//	int r3 = 2;
		
	dt *A1 = new dt[a*b*c]();	
	dt *A2 = new dt[a*b*c]();
	dt *A3 = new dt[a*b*c]();	//3 mode tensor to mat
	
/*	A1 = tensor2mat(A,a,b,c,1);		//a*bc
	A2 = tensor2mat(A,a,b,c,2);		//b*ac
	A3 = tensor2mat(A,a,b,c,3);		//c*ab  now we get 3 mode mats
*/
	Btensor2mat(A,A1,A2,A3,a,b,c);

//	msvd(A1,U1,a,b*c,r1);	//a*r1
//	msvd(A2,U2,b,a*c,r2);	//b*r2
//	msvd(A3,U3,c,a*b,r3);	//c*r3

	getvector(A1,U1,a,b*c,r1);
	getvector(A2,U2,b,a*c,r2);
	getvector(A3,U3,c,a*b,r3);
	//compute A ×1 U1'×2 U2'×3 U3'
	// first compute U1'[X1,X2,X3～～Xc]U2 = temp;   then temp*U3'   
	SBgemm(A,core,U1,U2,U3,a,b,c);
	


/*	cout<<"next to recover____________"<<endl;

	dt *rec = new dt[a*b*c]();		//store the recover Tensor	
	Recover(core,rec,U1,U2,U3,a,b,c);
	printTensor(rec,a,b,c);
	delete[] V;V=nullptr;
	delete[] rec; rec = nullptr;
*/

	delete[] A1; A1 = nullptr;
	delete[] A2; A2 = nullptr;
	delete[] A3; A3 = nullptr;

	}

// next is the function to return

void Recover(dt *core,dt *rec,dt *U1,dt *U2,dt *U3,int a ,int b,int c){
//	int r1 = 2;
//	int r2 = 3;
//	int r3 = 2;

	int r1 = 0.1*a;
	int r2 = 0.1*b;
	int r3 = 0.1*c;
//	dt *temp1 = new dt[r2*a*r3]();
	dt *temp2 = new dt[b*a*r3]();
	dt *temp3 = new dt[r3*a*b]();	//mode3 mat
	dt *temp = new dt[c*a*b]();	//result to be convert

	//compute A ×1 U1'×2 U2'×3 U3'
	// first compute U1'[X1,X2,X3～～Xc]U2 = temp;   then temp*U3'   
	//  U1 a*r1  U1' r1*a
	// A a*b*c
	// U2 b*r2  U2' r2*b
	// U3 c*r3  U3' r3*c

	dt alpha = 1.0;
	dt beta = 0.0;
	dt *d_core;
	dt *d_U1;
	dt *d_U2;
	dt *d_U3;
	dt *d_temp1;
	dt *d_temp2;
	dt *d_temp3;
	dt *d_temp;
	//
	cudaMalloc((void **)&d_core,r1*r2*r3*sizeof(dt));
	cudaMalloc((void **)&d_U1,a*r1*sizeof(dt));
	cudaMalloc((void **)&d_U2,b*r2*sizeof(dt));
	cudaMalloc((void **)&d_U3,c*r3*sizeof(dt));
	cudaMalloc((void **)&d_temp1,sizeof(dt)*r2*a*r3);
	cudaMalloc((void **)&d_temp2,sizeof(dt)*b*a*r3);
	cudaMalloc((void **)&d_temp3,sizeof(dt)*r3*a*b);	//mode 3 mat

	cudaMalloc((void **)&d_temp,sizeof(dt)*c*a*b);

	cudaMemcpy(d_core,core,sizeof(dt)*r1*r2*r3,cudaMemcpyHostToDevice);
	cudaMemcpy(d_U1,U1,sizeof(dt)*a*r1,cudaMemcpyHostToDevice);
	cudaMemcpy(d_U2,U2,sizeof(dt)*b*r2,cudaMemcpyHostToDevice);
	cudaMemcpy(d_U3,U3,sizeof(dt)*c*r3,cudaMemcpyHostToDevice);

	//cudaMemcpy(d_C,C,sizeof(dt)*a*d*c);

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasSgemmStridedBatched(
			
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			r2,				//row of A C
			a,				//col of B C
			r1,				//col of A ,row of B
			&alpha,
			d_core,
			r2,				//leading dimension store A
			r1*r2,			//step between two mat
		    d_U1,
			r1,
			0,
			&beta,
			d_temp1,
			r2,
			a*r2,
			r3				//batch number
			);
	//now d_temp1 store the real value col first
	
//	cudaMemcpy(temp1,d_temp1,sizeof(dt)*a*r2*r3,cudaMemcpyDeviceToHost);
//	printTensor(temp1,a,r2,r3);

	cublasSgemmStridedBatched(
			
			handle,
			CUBLAS_OP_T,
			CUBLAS_OP_N,
			b,
			a,
			r2,
			&alpha,
			d_U2,
			r2,
			0,
			d_temp1,
			r2,
			r2*a,
			&beta,
			d_temp2,
			b,
			b*a,
			r3
			);

	cudaMemcpy(temp2,d_temp2,sizeof(dt)*a*b*r3,cudaMemcpyDeviceToHost);
	
//	printTensor(temp2,a,b,r3);

	// now temp2 store the real value 
	//we will mat3,and the 
	temp3 = tensor2mat(temp2,a,b,r3,3);
//	printTensor(temp3,r3,a*b,1);
	

	cudaMemcpy(d_temp3,temp3,sizeof(dt)*r3*a*b,cudaMemcpyHostToDevice);

	cublasSgemm(
			
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			a*b,
			c,
			r3,
			&alpha,
			d_temp3,
			a*b,
			d_U3,
			r3,
			&beta,
			d_temp,
			a*b

			);

	cudaMemcpy(temp,d_temp,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost);

//	printTensor(temp,c,a*b,1);

	cublasDestroy(handle);
	
	//mode3 to Tensor 
	//r3 * r1×r2   to r1*r2*r3
	
	for(int k = 0;k<c;k++){
		for(int i = 0;i<a;i++){
			for(int j = 0;j<b;j++){
				rec[k*a*b+i*b+j] = temp[k*a*b+j*a+i];
			}
		}
	}

	cudaFree(d_core);
	cudaFree(d_U1);
	cudaFree(d_U2);
	cudaFree(d_U3);
	cudaFree(d_temp1);
	cudaFree(d_temp2);
	cudaFree(d_temp3);
	cudaFree(d_temp);
//	delete[] temp1; temp1 = nullptr;
	delete[] temp2; temp2 = nullptr;
	delete[] temp3; temp3 = nullptr;
	delete[] temp; temp = nullptr;

}



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

	cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);

	cublasGemmEx(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			m,
			m,
			n,
			&alpha,
			d_AT,
			CUDA_R_32F,
			m,
			d_A,
			CUDA_R_32F,
			n,
			&beta,
			d_AAT,  //store A*A'
			CUDA_R_32F,
			m,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT
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
void KRao(dt *X,dt *M,dt *N,dt *left,dt *right,int m,int n,int r,int k,int flag){
// m*r  n*r  m*n*r
	dt *d_M;
	cudaMalloc((void **)&d_M,sizeof(dt)*m*r);
	cudaMemcpy(d_M,M,sizeof(dt)*m*r,cudaMemcpyHostToDevice);
	
	dt *d_MT;
	cudaMalloc((void **)&d_MT,sizeof(dt)*m*r);
	dim3 threads(512,1,1);
	dim3 blocks1((m*r+512-1)/512,1,1);
	transpose<<<blocks1,threads>>>(d_M,d_MT,m,r);

	dt *d_MTM;
	cudaMalloc((void **)&d_MTM,sizeof(dt)*r*r);

	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;

	cublasCreate(&handle);
	cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
	cublasGemmEx(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			r,
			r,
			m,
			&alpha,
			d_M,
			CUDA_R_32F,
			r,
			d_MT,
			CUDA_R_32F,
			m,
			&beta,
			d_MTM,
			CUDA_R_32F,
			r,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT
			);
	cudaFree(d_MT);

	dt *d_N;
	dt *d_NT;
	cudaMalloc((void **)&d_N,sizeof(dt)*n*r);
	cudaMemcpy(d_N,N,sizeof(dt)*n*r,cudaMemcpyHostToDevice);

	cudaMalloc((void **)&d_NT,sizeof(dt)*n*r);
	dim3 blocks2((n*r+512-1)/512,1,1);
	transpose<<<blocks2,threads>>>(d_N,d_NT,n,r);
	

	//now d_MT*M  d_NT*N

	dt *d_NTN;
	cudaMalloc((void**)&d_NTN,sizeof(dt)*r*r);
	cublasGemmEx(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			r,
			r,
			n,
			&alpha,
			d_N,
			CUDA_R_32F,
			r,
			d_NT,
			CUDA_R_32F,
			n,
			&beta,
			d_NTN,
			CUDA_R_32F,
			r,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT
			);
	cudaFree(d_NT);

	dim3 blocks3((r*r+512-1)/512,1,1);
	elepro<<<blocks3,threads>>>(d_MTM,d_NTN,r);
	cudaMemcpy(right,d_MTM,sizeof(dt)*r*r,cudaMemcpyDeviceToHost);
	cudaFree(d_MTM);
	cudaFree(d_NTN);

	//right is solve the right

	dt *d_dot;
	cudaMalloc((void **)&d_dot,sizeof(dt)*m*n*r);
	dim3 blocks((m*n*r+512-1)/512,1,1);
	krpro<<<blocks,threads>>>(d_M,d_N,d_dot,m,n,r);
	cudaFree(d_M);
	cudaFree(d_N);
	//res store the dotpro  bc*a

	dt *d_X;
	dt *d_X_M;
	cudaMalloc((void**)&d_X,sizeof(dt)*m*n*k);
	cudaMalloc((void**)&d_X_M,sizeof(dt)*m*n*k);
	cudaMemcpy(d_X,X,sizeof(dt)*m*n*k,cudaMemcpyHostToDevice);

	dim3 blocks4((m*n*k+512-1)/512,1,1);
	if(flag == 1){
		mode1tran<<<blocks4,threads>>>(d_X,d_X_M,k,n,m);
	}else if(flag == 2){
		mode2tran<<<blocks4,threads>>>(d_X,d_X_M,n,k,m);
	}else{
		mode3tran<<<blocks4,threads>>>(d_X,d_X_M,n,m,k);
	}
	cudaFree(d_X);

	// d_X1*d_dot = left
	dt *d_left;
	cudaMalloc((void**)&d_left,sizeof(dt)*k*r);
	cublasGemmEx(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			r,
			k,
			m*n,
			&alpha,
			d_dot,
			CUDA_R_32F,
			r,
			d_X_M,
			CUDA_R_32F,
			m*n,
			&beta,
			d_left,
			CUDA_R_32F,
			r,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT
			);
	cudaMemcpy(left,d_left,sizeof(dt)*k*r,cudaMemcpyDeviceToHost);
	cublasDestroy(handle);
	cudaFree(d_left);
	cudaFree(d_X_M);
	cudaFree(d_dot);

}



void solve(dt *left,dt *right,dt *res,int r,int m){
	dt *d_left;
	dt *d_right;
	cudaMalloc((void**)&d_right,sizeof(dt)*m*r);
	cudaMalloc((void**)&d_left,sizeof(dt)*r*r);
	dt *d_work;
	int *d_info;
	int lwork;
	cusolverDnHandle_t handle;
	cusolverDnCreate(&handle);
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	cudaMalloc((void**)&d_info,sizeof(int));

	cudaMemcpy(d_left,left,sizeof(dt)*r*r,cudaMemcpyHostToDevice);
	cudaMemcpy(d_right,right,sizeof(dt)*m*r,cudaMemcpyHostToDevice);
	cusolverDnSpotrf_bufferSize(
			handle,
			uplo,
			r,
			d_left,
			r,
			&lwork
			);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);

	cusolverDnSpotrf(
			handle,
			uplo,
			r,
			d_left,
			r,
			d_work,
			lwork,
			d_info
			);
	cusolverDnSpotrs(
			handle,
			uplo,
			r,
			m,
			d_left,
			r,
			d_right,
			r,
			d_info
			);
	cudaDeviceSynchronize();
//	int info_gpu;
//	cudaMemcpy(&info_gpu,d_info,sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(res,d_right,sizeof(dt)*m*r,cudaMemcpyDeviceToHost);
/*	if(info_gpu == 0){
		cout<<"OK"<<endl;
		cout<<endl;
	}
*/
// d_right store the A/B/C m*r


//	printTensor(res,m,r,1);
	dt *sum = new dt[r]();
	for(int i = 0;i<r;i++){
		for(int j = 0;j<m;j++){
				sum[i] += res[j*r+i]*res[j*r+i]; 
			}
		}

//	printTensor(sum,r,1,1);
	dt *d_sum;
	cudaMalloc((void**)&d_sum,sizeof(dt)*r);
	cudaMemcpy(d_sum,sum,sizeof(dt)*r,cudaMemcpyHostToDevice);
	dim3 threads(512,1,1);
	dim3 blocks((m*r+512-1)/512,1,1);
	norm<<<blocks,threads>>>(d_right,d_sum,m,r);
	cudaMemcpy(res,d_right,sizeof(dt)*m*r,cudaMemcpyDeviceToHost);
	cudaFree(d_sum);
	cudaFree(d_left);
	cudaFree(d_right);
	cudaFree(d_info);
	cudaFree(d_work);
	cusolverDnDestroy(handle);
	cudaDeviceReset();
	delete[] sum;sum = nullptr;


}

/*dt norm(dt *X,int a,int b,int c){
	dt temp = 0.0;
	dt *d_X;
	dt d_temp;
	cudaMalloc(d_X,sizeof(dt)*a*b*c);
	cudaMalloc(d_temp,sizeof(dt));
	cudaMemcpy(d_X,X,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);
	dim3 threads(512,1,1);
	dim3 blocks((a*b*c+512-1)/512,1,1);
	Norm<<<blocks,threads>>>(d_X,d_temp,a,b,c);
	cudaMemcpy(temp,d_temp)

	
}*/

/*void recontr(dt *src,dt *des,dt *A,dt *B,dt *C,int a,int b,int c,int r){
	dt *d_C;
	dt *d_B;
	dt *d_A;
	dt *d_des;
	dt *d_src;
	dt *d_BA;

	cudaMalloc(d_B,sizeof(dt)*b*r);
	cudaMalloc(d_A,sizeof(dt)*a*r);
	cudaMalloc(d_CB,sizeof(dt)*a*b*r);
	cudaMemcpy(d_B,B,sizeof(dt)*b*r);
	cudaMemcpy(d_A,A,sizeof(dt)*a*r);


}*/

void cp_als(dt *X,dt *A,dt *B,dt *C,int a,int b,int c,int r){

/*	dt *X_temp = new dt[a*b*c]();
	dt *error = new dt[100];
	dt tol = 1e06;
	dt X_norm = 0.0;
	for(int i = 0;i<a*b*c;i++){
		X_norm += X[i]*X[i];
	}
	X_norm = sqrt(X_norm);
*/
	dt *temp1 = new dt[a*r]();
	dt *temp2 = new dt[b*r]();
	dt *temp3 = new dt[c*r]();
	dt *tem1 = new dt[r*r]();
	dt *tem2 = new dt[r*r]();
	dt *tem3 = new dt[r*r]();
	
	for(int i = 0;i<1;i++){

		KRao(X,C,B,temp1,tem1,c,b,r,a,1);
		solve(tem1,temp1,A,r,a);     // we get A  

		KRao(X,C,A,temp2,tem2,c,a,r,b,2);
		solve(tem2,temp2,B,r,b);     // we get B
		
		KRao(X,B,A,temp3,tem3,b,a,r,c,3);
		solve(tem3,temp3,C,r,c);    //we get C

//		recontr(X,X_temp,A,B,C,a,b,c,r);

//		error[i] = 
	}

	delete[] temp1;temp1 = nullptr;
	delete[] temp2;temp1 = nullptr;
	delete[] temp3;temp1 = nullptr;
	delete[] tem1;tem1 = nullptr;
	delete[] tem2;tem2 = nullptr;
	delete[] tem3;tem3 = nullptr;
}
