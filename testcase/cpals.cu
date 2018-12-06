#include "head.h"

void allin(dt *X,dt *M,dt *N,dt *res,int m,int n,int r,int k,int flag){
	// X m*n*k left  m*r right  n*r 
	dt *d_X;
	cudaMalloc((void**)&d_X,sizeof(dt)*m*n*k);
	dt *d_X_M;
	cudaMalloc((void**)&d_X_M,sizeof(dt)*m*n*k);
	cudaMemcpy(d_X,X,sizeof(dt)*m*n*k,cudaMemcpyHostToDevice);

	dim3 thread(512,1,1);
	dim3 block((m*n*k+512-1)/512,1,1);
	if(flag == 1){
		mode1tran<<<block,thread>>>(d_X,d_X_M,k,n,m);
	}else if(flag == 2){
		mode2tran<<<block,thread>>>(d_X,d_X_M,n,k,m);
	}else{
		mode3tran<<<block,thread>>>(d_X,d_X_M,n,m,k);
	}
	
	cudaFree(d_X);  //d_X_M(flag)
	// now we compute M'*M and N'*N

	dt *d_M;
	cudaMalloc((void**)&d_M,sizeof(dt)*m*r); //store with row
	cudaMemcpy(d_M,M,sizeof(dt)*m*r,cudaMemcpyHostToDevice);
	dt *d_MM;
	cudaMalloc((void**)&d_MM,sizeof(dt)*r*r); 

	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	cublasSsyrk(
		handle,
		uplo,
		CUBLAS_OP_N,
		r,m,
		&alpha,
		d_M,r,
		&beta,
		d_MM,r	
		);
//	cudaFree(d_M);
	//d_MM is store in half 

	dt *d_N;
	cudaMalloc((void**)&d_N,sizeof(dt)*n*r);
	cudaMemcpy(d_N,N,sizeof(dt)*n*r,cudaMemcpyHostToDevice);
	dt *d_NN;
	cudaMalloc((void**)&d_NN,sizeof(dt)*r*r);

	cublasSsyrk(
		handle,
		uplo,
		CUBLAS_OP_N,
		r,n,
		&alpha,
		d_N,r,
		&beta,
		d_NN,r	
		);
//	cudaFree(d_N);

	dim3 block1((r*r+512-1),1,1);
	elepro<<<block1,thread>>>(d_MM,d_NN,r);
	//d_MM store the half element
/*	dt *tt = new dt[r*r]();
	cudaMemcpy(tt,d_MM,sizeof(dt)*r*r,cudaMemcpyDeviceToHost);
	printTensor(tt,r,r,1);
	delete[] tt;tt=nullptr;
	cout<<"this is yuansucheng"<<endl;
*/
	dt *d_dot;
	cudaMalloc((void **)&d_dot,sizeof(dt)*m*n*r);
	dim3 block2((m*n*r+512-1)/512,1,1);
	krpro<<<block2,thread>>>(d_M,d_N,d_dot,m,n,r);
	cudaFree(d_M);
	cudaFree(d_N);
/*	dt *t1 = new dt[m*n*r]();
	cudaMemcpy(t1,d_dot,sizeof(dt)*m*n*r,cudaMemcpyDeviceToHost);
	printTensor(t1,m*n,r,1);
	delete[] t1;t1=nullptr;
	cout<<"this is KR product"<<endl;
*/
	// d_X_M is k*mn d_dot mn*r store with row
	// d_X_M * d_dot

	dt *d_req;
	cudaMalloc((void**)&d_req,sizeof(dt)*r*k);
	cublasSgemm(

		handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		r,k,m*n,
		&alpha,
		d_dot,r,
		d_X_M,m*n,
		&beta,
		d_req,r
			);
	cudaFree(d_X_M);
	cudaFree(d_dot);
	cublasDestroy(handle);

/*	dt *t2 = new dt[k*r]();
	cudaMemcpy(t2,d_req,sizeof(dt)*k*r,cudaMemcpyDeviceToHost);
	printTensor(t2,k,r,1);
	delete[] t2;t2=nullptr;
	cout<<"this is left"<<endl;
*/
	//d_req  is the left k*r
	// d_MM is the right r*r 
	// the result is k*r

	dt *d_work;
	int *d_info;
	int lwork;
	cusolverDnHandle_t handle1;
	cusolverDnCreate(&handle1);
	cudaMalloc((void**)&d_info,sizeof(int));

	cusolverDnSpotrf_bufferSize(
			handle1,
			uplo,
			r,
			d_MM,
			r,
			&lwork
			);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);

	cusolverDnSpotrf(
			handle1,
			uplo,
			r,
			d_MM,
			r,
			d_work,
			lwork,
			d_info
			);
	cusolverDnSpotrs(
			handle1,
			uplo,
			r,
			k,
			d_MM,
			r,
			d_req,
			r,
			d_info
			);
	cudaDeviceSynchronize();
//	int info_gpu;
//	cudaMemcpy(&info_gpu,d_info,sizeof(int),cudaMemcpyDeviceToHost);

	cudaMemcpy(res,d_req,sizeof(dt)*k*r,cudaMemcpyDeviceToHost);
//	printTensor(res,k,r,1);
	cudaFree(d_MM);

	dt *sum = new dt[r]();
	for(int i = 0;i<r;i++){
		for(int j = 0;j<k;j++){
				sum[i] += res[j*r+i]*res[j*r+i]; 
			}
		}

//	printTensor(sum,r,1,1);
	dt *d_sum;
	cudaMalloc((void**)&d_sum,sizeof(dt)*r);
	cudaMemcpy(d_sum,sum,sizeof(dt)*r,cudaMemcpyHostToDevice);
	dim3 block4((k*r+512-1)/512,1,1);
	norm<<<block4,thread>>>(d_req,d_sum,k,r);
	cudaMemcpy(res,d_req,sizeof(dt)*k*r,cudaMemcpyDeviceToHost);
	cudaFree(d_sum);
	delete[] sum;sum=nullptr;

	cudaFree(d_req);
	cudaFree(d_info);
	cudaFree(d_work);
	cusolverDnDestroy(handle1);
	cudaDeviceReset();

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
	half *d_m;
	half *d_mt;
	cudaMalloc((void **)&d_m,sizeof(half)*m*r);
	cudaMalloc((void **)&d_mt,sizeof(half)*m*r);
	f2h<<<blocks1,threads>>>(d_M,d_m,m*r);
	f2h<<<blocks1,threads>>>(d_MT,d_mt,m*r);
	cudaFree(d_MT);

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
			d_m,
			CUDA_R_16F,
			r,
			d_mt,
			CUDA_R_16F,
			m,
			&beta,
			d_MTM,
			CUDA_R_32F,
			r,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT
			);
	cudaFree(d_m);
	cudaFree(d_mt);



	dt *d_N;
	dt *d_NT;
	cudaMalloc((void **)&d_N,sizeof(dt)*n*r);
	cudaMemcpy(d_N,N,sizeof(dt)*n*r,cudaMemcpyHostToDevice);

	cudaMalloc((void **)&d_NT,sizeof(dt)*n*r);
	dim3 blocks2((n*r+512-1)/512,1,1);
	transpose<<<blocks2,threads>>>(d_N,d_NT,n,r);
	half *d_n;
	half *d_nt;
	cudaMalloc((void **)&d_n,sizeof(half)*n*r);
	cudaMalloc((void **)&d_nt,sizeof(half)*n*r);
	
	f2h<<<blocks2,threads>>>(d_N,d_n,n*r);
	f2h<<<blocks2,threads>>>(d_NT,d_nt,n*r);
	cudaFree(d_NT);

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
			d_n,
			CUDA_R_16F,
			r,
			d_nt,
			CUDA_R_16F,
			n,
			&beta,
			d_NTN,
			CUDA_R_32F,
			r,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT
			);
	cudaFree(d_n);
	cudaFree(d_nt);

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

	half *d_x_m;
	half *d_hdot;
	cudaMalloc((void**)&d_x_m,sizeof(half)*m*n*k);
	cudaMalloc((void**)&d_hdot,sizeof(half)*m*n*r);
	f2h<<<blocks4,threads>>>(d_X_M,d_x_m,m*n*k);
	f2h<<<blocks,threads>>>(d_dot,d_hdot,m*n*r);

	cudaFree(d_X_M);
	cudaFree(d_dot);

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
			d_hdot,
			CUDA_R_16F,
			r,
			d_x_m,
			CUDA_R_16F,
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
	cudaFree(d_x_m);
	cudaFree(d_hdot);
	cudaFree(d_left);

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
    clock_t lu,guo,be,yi;
    lu =clock();
	dt *temp1 = new dt[a*r]();
	dt *temp2 = new dt[b*r]();
	dt *temp3 = new dt[c*r]();
	dt *tem1 = new dt[r*r]();
	dt *tem2 = new dt[r*r]();
	dt *tem3 = new dt[r*r]();
	guo = clock();
	cout<<"分配空间"<<"  ";
	cout<<(double)(guo-lu)/CLOCKS_PER_SEC<<"s"<<endl;
	
	for(int i = 0;i<1;i++){
        
        clock_t t1,t2,t3;
        t1 = clock();
		KRao(X,C,B,temp1,tem1,c,b,r,a,1);
        t2 = clock();
    cout<<"计算左右值"<<"  "; 
	cout<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<endl;
      
		solve(tem1,temp1,A,r,a);     // we get A  
    t3 = clock();
    cout<<"zuizhongzhi"<<"  "; 
	cout<<(double)(t3-t2)/CLOCKS_PER_SEC<<"s"<<endl;
    

		KRao(X,C,A,temp2,tem2,c,a,r,b,2);
		solve(tem2,temp2,B,r,b);     // we get B
		
		KRao(X,B,A,temp3,tem3,b,a,r,c,3);
		solve(tem3,temp3,C,r,c);    //we get C

//		recontr(X,X_temp,A,B,C,a,b,c,r);

//		error[i] = 
	}
    yi = clock();
	delete[] temp1;temp1 = nullptr;
	delete[] temp2;temp1 = nullptr;
	delete[] temp3;temp1 = nullptr;
	delete[] tem1;tem1 = nullptr;
	delete[] tem2;tem2 = nullptr;
	delete[] tem3;tem3 = nullptr;
	be = clock();
	cout<<"释放空间"<<"  ";
	cout<<(double)(be-yi)/CLOCKS_PER_SEC<<"s"<<endl;
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
	*/
