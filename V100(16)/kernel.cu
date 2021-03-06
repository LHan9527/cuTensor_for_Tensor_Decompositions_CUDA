#include "opera.h"

__global__ void transpose(dt *A,dt* AT,int m,int n){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i<n*m){
		int row = i/n;
		int col = i%n;
		AT[col*m+row] = A[row*n+col];
		i+=temp;
	}
	__syncthreads();
}

__global__ void mode1tran(dt *AA,dt *A1,int a,int b,int c){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i<a*b*c){
		int tube = i/(a*b);		//which slice	
		int row = (i-tube*(a*b))/b;	
		int col = (i-tube*(a*b))%b;		//get the index
		A1[row*b*c+tube*b+col] = AA[tube*a*b+row*b+col];
		i+=temp;
	}
	__syncthreads();
}

__global__ void mode2tran(dt *AA,dt *A2,int a,int b,int c){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i<a*b*c){
		int tube = i/(a*b);		//which slice	
		int row = (i-tube*(a*b))/b;	
		int col = (i-tube*(a*b))%b;		//get the index
		A2[col*a*c+tube*a+row] = AA[tube*a*b+row*b+col];
		i+=temp;
	}
	__syncthreads();
}

__global__ void mode3tran(dt *AA,dt *A3,int a,int b,int c){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i<a*b*c){
		int tube = i/(a*b);		//which slice	
		int row = (i-tube*(a*b))/b;	
		int col = (i-tube*(a*b))%b;		//get the index
		A3[tube*a*b+col*a+row] = AA[tube*a*b+row*b+col];
		i+=temp;
	}
	__syncthreads();
}
__global__ void tran3mode(dt *AA,dt *A3,int a,int b,int c){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i<a*b*c){
		int tube = i/(a*b);		//which slice	
		int row = (i-tube*(a*b))/b;	
		int col = (i-tube*(a*b))%b;		//get the index
	//	A3[tube*a*b+col*a+row] = AA[tube*a*b+row*b+col];
		A3[tube*a*b+row*b+col] = AA[tube*a*b+col*a+row];
		i+=temp;
	}
	__syncthreads();
}


__global__ void krpro(dt *M,dt *N,dt *res,int m,int n,int r){
	
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i<m*n*r){
		int row = i/r;
		int col = i%r;
		res[row*r+col] = M[(row/n)*r+col]*N[(row%n)*r+col];
		i+=temp;
	}
	__syncthreads();
}

__global__ void elepro(dt *A,dt *B,int r){
	
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i<r*r){
		A[i] = A[i]*B[i];
		i+=temp;
	}
	__syncthreads();
}

__global__ void norm(dt *A,dt *B,int m ,int r){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i<m*r){
		int row = i/r;
		int col = i%r;
		A[row*r+col] = A[row*r+col]/sqrt(B[col]);
		i+=temp;
	}
	__syncthreads();
}

__global__ void f2h(dt *A,half *B,int m){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i<m){
		B[i] = __float2half(A[i]);
		i+=temp;
	}
	__syncthreads();
}

