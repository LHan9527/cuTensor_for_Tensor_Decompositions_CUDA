#include "head.h"


void setMatrice(dt *A,int m,int n){
	for(int i = 0;i<m*n;i++){
		A[i] = rand()*0.1/(RAND_MAX*0.1);
	}
}
int main(int args,char *argv[]){
	int a = atoi(argv[1]);
	int b = atoi(argv[2]);
	int c = atoi(argv[3]);

	clock_t t1,t2,t3,t4;
	t1=clock();
	dt **h_A = new dt*[c];
	dt **h_B = new dt*[c]; //host CPU
	dt **h_C = new dt*[c];
	dt **h_result = new dt*[c];
	for(int i = 0;i<c;i++){
		h_result[i] = new dt[a*a];
		cudaMalloc((void**)&h_A[i],sizeof(dt)*a*b);
		cudaMalloc((void**)&h_B[i],sizeof(dt)*b*a);
		cudaMalloc((void**)&h_C[i],sizeof(dt)*a*a);
		// malloc memory on GPU and link to host
	}//link device to host

	dt **d_A;
	dt **d_B;
	dt **d_C;
	cudaMalloc((void**)&d_A,sizeof(*h_A)*c);
	cudaMalloc((void**)&d_B,sizeof(*h_B)*c);
	cudaMalloc((void**)&d_C,sizeof(*h_C)*c);

	cudaMemcpy(d_A,h_A,sizeof(*h_A)*c,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,h_B,sizeof(*h_B)*c,cudaMemcpyHostToDevice);
	cudaMemcpy(d_C,h_C,sizeof(*h_C)*c,cudaMemcpyHostToDevice);
	//Device malloc memory and transfer Host to Device

	dt *A = new dt[a*b]();
	dt *B = new dt[b*a]();
	dt *C = new dt[a*a]();
	for(int i =0;i<c;i++){
		setMatrice(A,a,b);
//		printTensor(A,a,b,1);
		cublasSetMatrix(a,b,sizeof(dt),A,a,h_A[i],a);
		setMatrice(B,b,a);
//		printTensor(B,b,a,1);
		cublasSetMatrix(b,a,sizeof(dt),B,b,h_B[i],b);
	}

	const dt **AA =(const dt **)d_A;
	const dt **BB =(const dt **)d_B;
	t3 = clock();
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemmBatched(
		handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		a,
		a,
		b,
		&alpha,
		BB,
		a,
		AA,
		b,
		&beta,
		d_C,
		a,
		c
		);
	t4 = clock();
	for(int i = 0;i<c;i++){
		cublasGetMatrix(a,a,sizeof(dt),h_C[i],a,h_result[i],a);
//		printTensor(h_result[i],a,a,1);
	}
	cublasDestroy(handle);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	delete[] h_A;h_A = NULL;
	delete[] h_B;h_B = NULL;
	delete[] h_C;h_C = NULL;
	delete[] h_result;h_result = NULL;
	t2 = clock();
	cout<<(double)(t4-t3)/CLOCKS_PER_SEC<<"s"<<endl;
	cout<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<endl;
	return 0;
}

