
#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <cublas_v2.h>


// #include <blast/batch_gemm.h>
// #include <blast/system/cublas/execution_policy.h>

// #include "mkl_trans.h"
// #include "cublas_batch_gemm.cuh"
using namespace std;
using namespace Eigen;



int main()
{
  
  float time_1 = 0, time_2 = 0, time_3 = #if (defined __GNUC__) && (__GNUC__>4 || __GNUC_MINOR__>=7)
  #undef _GLIBCXX_ATOMIC_BUILTINS
  #undef _GLIBCXX_USE_INT128
#endif0, time_4 = 0, time_0 = 0;
  cublasHandle_t handle; cublasCreate(&handle);
  ofstream fout("gputiming_float_kernel2.txt",ios::out|ios::app);
  fout << "t_size" <<'\t'<< "k_size"<<'\t'<< "iter_max"<<'\t' << "time_1" << '\t' << "time_2" << '\t' << "time_3" << '\t' << "time_4"<< '\n';

  int t_size = 120; int k_size = 10;
  int iter_max = 200;    
    
  const int I = 120;
  //   int *ptrI; ptrI = (int*)(&I); *ptrI = 20; cout << I <<endl;
  const int J = 120;
  //  int *ptrJ; ptrJ = (int*)(&J);*ptrJ = t_size;
  const int K = 120;
  //  int *ptrK; ptrK = (int*)(&K);*ptrK = t_size;
  
  const int Q = 10; const int P = 10; const int R = 10;
  size_t sd = sizeof(float);
  // float X[I*J*K];
  size_t chars_read = 0;
  float *X;
  X = (float*)malloc(I*J*K*sd);
  float *d_X; cudaMalloc((void **)&d_X, I*J*K*sd);
  
  ifstream input("/home/yang/btas_test_2/dataX120_10.txt");
  while (input && chars_read <I*J*K)
  {
    input >>X[chars_read];
    chars_read++;
  }
  cublasSetVector(I*J*K,sd,X,1,d_X,1);
  //  h_X = X;
   
   /*  Tensor<float> T(I,J,K); int idx = 0;
   for(int k=0; k<K; k++){
       for(int j=0;j<J;j++){
           for(int i=0;i<I;i++){
               T(i,j,k) = X[idx];
               idx = idx+1;
            }
        }
	}*/
  //////////////////////////////////////////////////////////////////////////// 
 ////////////////////////// Initial A1 A2 A3 ////////////////////////////////
///////////////    T_IJK = [G_PQR;A1_IP,A2_JQ,A3_KR]  /////////////////////////
 /////////////////////////////////////////////////////////////////////////////
 /* reshape tensor T_IJK along different mode */
  


  MatrixXf X1 = MatrixXf::Zero(I,J*K);
   MatrixXf X2 = MatrixXf::Zero(J,I*K);
   MatrixXf X3 = MatrixXf::Zero(K,I*J);
    
   for(int i = 0; i<K; ++i){
     float slice[I*J];int m = 0;
     for(int ii = 0; ii<I*J; ++ii){
       slice[m] = float(X[I*J*i+ii]);
       m++;
     }
     MatrixXf test1 =  Map<MatrixXf,0, Stride<I,1> >(slice,I,J,Stride<I,1>(I,1));
     X1.block<I,J>(0,J*i) =  test1;
   } 

   for(int i = 0; i<K; ++i){
     float slice[I*J];int m = 0;
     for(int ii = 0; ii<I*J; ++ii){
       slice[m] = float(X[I*J*i+ii]);
       m++;
     }
     MatrixXf test2 =  Map<MatrixXf,0, Stride<1,I> >(slice,J,I,Stride<1,I>(1,I));
     X2.block<J,I>(0,I*i) =  test2;
   } 

   for(int i = 0; i<J; ++i){
     float slice[I*K];int m = 0;
     for(int ii = 0; ii<K; ++ii){
       for (int jj = 0; jj<I; ++jj){
       slice[m] = float(X[I*i+I*J*ii+jj]);
       m++;
       }
     }
     MatrixXf test3 =  Map<MatrixXf,0, Stride<1,I> >(slice,K,I,Stride<1,I>(1,I));
     X3.block<K,I>(0,I*i) =  test3;
     
   } 
   //  cout << X3 << endl;
   //  cout <<X3.rows()
   /*initialize A1, A2, A3 */
 

   MatrixXf A1 = MatrixXf::Zero(I,P);
   MatrixXf A2 = MatrixXf::Zero(J,Q);
   MatrixXf A3 = MatrixXf::Zero(K,R);
   
   JacobiSVD<MatrixXf> svd1(X1,ComputeThinU);
   const Eigen::MatrixXf U1 = svd1.matrixU();
   A1 = U1.block<I,P>(0,0);
   
   JacobiSVD<MatrixXf> svd2(X2,ComputeThinU);
   const Eigen::MatrixXf U2 = svd2.matrixU();
   A2 = U2.block<J,Q>(0,0);
   
   JacobiSVD<MatrixXf> svd3(X3,ComputeThinU);
   const Eigen::MatrixXf U3 = svd3.matrixU();
   A3 = U3.block<K,R>(0,0);
   cout << "Initial A1,A2,A3:" << endl;
   //  cout << A1 << endl;
   //  cout << A2 << endl;
   // cout << A3 << endl;
   float *h_A1; h_A1 = (float*)malloc(I*P*sd);
   Map<MatrixXf>(h_A1,A1.rows(),A1.cols())= A1;  
   float *h_A2; h_A2 = (float*)malloc(J*Q*sd);
   Map<MatrixXf>(h_A2,A2.rows(),A2.cols())= A2;   
   float *h_A3; h_A3 = (float*)malloc(K*R*sd);
   Map<MatrixXf>(h_A3,A3.rows(),A3.cols())= A3;
   float *d_A1, *d_A2, *d_A3;
   cudaMalloc((void **)&d_A1, I*P*sd);
   cudaMalloc((void **)&d_A2, J*Q*sd);
   cudaMalloc((void **)&d_A3, K*R*sd);
   cublasSetVector(I*P,sd,h_A1,1,d_A1,1);
   cublasSetVector(J*Q,sd,h_A2,1,d_A2,1);
   cublasSetVector(K*R,sd,h_A3,1,d_A3,1);
  ///////////////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////Using Kernel to do Tensor Contraction ////////////////////
  MatrixXf Y1 = MatrixXf::Random(I,Q*R);
  MatrixXf Y2 = MatrixXf::Random(J,P*R);
  MatrixXf Y3 = MatrixXf::Random(K,P*Q);
  float *Y1_eigen; Y1_eigen =(float*)malloc(I*Q*R*sd); float *d_Y1_eigen;
  float *Y2_eigen;Y2_eigen =(float*)malloc(P*J*R*sd); float *d_Y2_eigen;
  float *Y3_eigen;Y3_eigen =(float*)malloc(P*Q*K*sd); float *d_Y3_eigen;
  float alpha = 1.0f, beta = 0.0f;
  float *temp_1; temp_1 = (float*)malloc(I*Q*K*sd); float *d_temp_1;
  float *temp_2; temp_2 = (float*)malloc(I*J*R*sd); float *d_temp_2;
  float *temp_3; temp_3 = (float*)malloc(P*J*K*sd); float *d_temp_3;
  cudaMalloc((void **)&d_Y1_eigen, I*Q*R*sd);
  cudaMalloc((void **)&d_Y2_eigen, P*J*R*sd);
  cudaMalloc((void **)&d_Y3_eigen,P*Q*K*sd);
  cudaMalloc((void **)&d_temp_1,I*Q*K*sd);
  cudaMalloc((void **)&d_temp_2,I*J*R*sd);
  cudaMalloc((void **)&d_temp_3,P*J*K*sd);
  /////////////////// UPDATES ////////////////////////////////////////////////
  cudaEvent_t start, stop;
  for (int iter = 0; iter <iter_max; iter++){
    cout << "iter no:" << iter <<endl;
    ///////////////////////////////////////////////////////////////
    /*    UPDATE A1    */ 
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,I,Q,J,&alpha,d_X,I,I*J,d_A2,J,0,&beta,d_temp_1,I,I*Q,K);
    cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,I,Q,K,&alpha,d_temp_1,I,I*Q,d_A3,K,0,&beta,d_Y1_eigen,I,I*Q,R);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_0, start,stop); time_1 += (float)time_0;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasGetVector(I*Q*R,sd,d_Y1_eigen,1,Y1_eigen,1);
    //  Y1_eigen = BTAS_Contract1(T,A2,A3,I,J,K,Q,R);
    for(int i = 0; i<R; ++i){
      float slice[I*Q];int m = 0;
      for(int ii = 0; ii<I*Q; ++ii){
	slice[m] = float(Y1_eigen[I*Q*i+ii]);
	m++;
      }
      MatrixXf test1 =  Map<MatrixXf,0, Stride<I,1> >(slice,I,Q,Stride<I,1>(I,1));
      Y1.block<I,Q>(0,Q*i) =  test1;
    } 
// 	cout << "Y1 is:"<<endl; 
    //   cout << Y1 << endl;
    
    JacobiSVD<MatrixXf> svda1(Y1,ComputeThinU);
    const Eigen::MatrixXf Ua1 = svda1.matrixU();
    A1 = Ua1.block<I,P>(0,0);
    //  cout << "A1 IS:"<<endl;
    //  cout << A1 << endl;
    Map<MatrixXf>(h_A1,A1.rows(),A1.cols())= A1;  
    cublasSetVector(I*P,sd,h_A1,1,d_A1,1);
////////////////////////////////////////////////////////////////////////////////////////////   
     /*    UPDATE A2    */ 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cublasSgemmBatched(handle, CUBLAS_OP_N,CUBLAS_OP_N,I,J,K,&alpha,d_X,I,I*J,d_A3,K,0,&beta,d_temp_2,I,I*J,R);
    cublasSgemmBatched(handle, CUBLAS_OP_T,CUBLAS_OP_N,P,J,I,&alpha,d_A1,I,0,d_temp_2,I,I*J,&beta,d_Y2_eigen,P,P*J,R);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_0, start,stop); time_2 += (float)time_0;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasGetVector(P*J*R,sd,d_Y2_eigen,1,Y2_eigen,1);
  

//      Y2_eigen = BTAS_Contract2(T,A1,A3,I,J,K,P,R);
    for(int i = 0; i<R; ++i){
      float slice[P*J];int m = 0;
      for(int ii = 0; ii<P*J; ++ii){
	slice[m] = float(Y2_eigen[P*J*i+ii]);
	m++;
      }
      MatrixXf test2 =  Map<MatrixXf,0, Stride<1,P> >(slice,J,P,Stride<1,P>(1,P));
      Y2.block<J,P>(0,P*i) =  test2;
    } 
// 	cout <<"Y2 is:"<<endl;  
    //   cout<< Y2 << endl;
    JacobiSVD<MatrixXf> svda2(Y2,ComputeThinU);
    const Eigen::MatrixXf Ua2 = svda2.matrixU();
    A2 = Ua2.block<J,Q>(0,0);
    // cout << "A2 IS:"<<endl;
    // cout << A2 <<endl;
    Map<MatrixXf>(h_A2,A2.rows(),A2.cols())= A2;  
    cublasSetVector(J*Q,sd,h_A2,1,d_A2,1);
/////////////////////////////////////////////////////////////////////////////////////////////
    /*    UPDATE A3    */ 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    cublasSgemmBatched(handle, CUBLAS_OP_T,CUBLAS_OP_N,P,J,I,&alpha,d_A1,I,0,d_X,I,I*J,&beta,d_temp_3,P,P*J,K);
    cublasSgemmBatched(handle, CUBLAS_OP_N,CUBLAS_OP_T,P,Q,J,&alpha,d_temp_3,P,P*J,d_A2,J,0,&beta,d_Y3_eigen,P,P*Q,K);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_0, start,stop); time_3 += (float)time_0;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasGetVector(P*Q*K,sd,d_Y3_eigen,1,Y3_eigen,1);
    
    
    //  Y3_eigen = BTAS_Contract3(T,A1,A2,I,J,K,P,Q);
    for(int i = 0; i<Q; ++i){
      float slice[P*K];int m = 0;
      for(int ii = 0; ii<K; ++ii){
	for (int jj = 0; jj<P; ++jj){
	  slice[m] = float(Y3_eigen[P*i+P*Q*ii+jj]);
	  m++;
	}
      }
      MatrixXf test3 =  Map<MatrixXf,0, Stride<1,P> >(slice,K,P,Stride<1,P>(1,P));
      Y3.block<K,P>(0,P*i) =  test3;
    } 
    //  cout << "Y3 IS:" << endl;    
    //   cout<< Y3 << endl;
    
    JacobiSVD<MatrixXf> svda3(Y3,ComputeThinU);
    const Eigen::MatrixXf Ua3 = svda3.matrixU();
    A3 = Ua3.block<K,R>(0,0);
    //  cout <<"A3 IS:"<<endl;
    //  cout << A3 <<endl;	
    Map<MatrixXf>(h_A3,A3.rows(),A3.cols())= A3;  
    cublasSetVector(K*R,sd,h_A3,1,d_A3,1);	
  }
  ///////////////////////////////////////////////////////////////////////
///////////////////// UPDATE G //////////////////////////////////////// 
  
  float *temp_4; temp_4 = (float*)malloc(P*J*K*sd); float *d_temp_4; cudaMalloc((void **)&d_temp_4,P*J*K*sd);
  float *G; G = (float*)malloc(P*Q*R*sd); float *d_G; cudaMalloc((void **)&d_G,P*Q*R*sd);
  
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  
  
  cublasSgemmBatched(handle, CUBLAS_OP_T,CUBLAS_OP_N,P,J,I,&alpha,d_A1,I,0,d_X,I,I*J,&beta,d_temp_3,P,P*J,K);
  cublasSgemmBatched(handle, CUBLAS_OP_N,CUBLAS_OP_N,P,J,K,&alpha,d_temp_3,P,P*J,d_A3,K,0,&beta,d_temp_4,P,P*J,R);
  cublasSgemmBatched(handle, CUBLAS_OP_N,CUBLAS_OP_N,P,Q,J,&alpha,d_temp_4,P,P*J,d_A2,J,0,&beta,d_G,P,P*Q,R);
 
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_0, start,stop); time_4 += (float)time_0;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasGetVector(P*Q*R,sd,d_G,1,G,1); 
  
  fout << t_size <<'\t' << k_size << '\t' << iter_max<< '\t' << time_1 << '\t' << time_2  << '\t' << time_3 << '\t' << time_4 << '\n';
  
  
  
  
  free(X); free(h_A1);free(Y1_eigen); free(h_A2);free(Y2_eigen); free(h_A3);free(Y3_eigen); free(G);
  free(temp_1);free(temp_2);free(temp_3); free(temp_4);
  
  cudaFree(d_A1);cudaFree(d_A2);cudaFree(d_A3);cudaFree(d_temp_1);cudaFree(d_temp_2);cudaFree(d_temp_3);cudaFree(d_temp_4);cudaFree(d_Y1_eigen);cudaFree(d_Y2_eigen);cudaFree(d_Y3_eigen);
  cudaFree(d_G);
  cublasDestroy(handle);
  fout.close();
  return 0;
}

