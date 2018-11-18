
#include "head.h"
#include "Eigen/Dense"


// #include <blast/batch_gemm.h>
// #include <blast/system/cublas/execution_policy.h>

// #include "mkl_trans.h"
// #include "cublas_batch_gemm.cuh"
using namespace std;
using namespace Eigen;



int main()
{
    
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
  for(int i = 0;i<I*J*K;i++){
     X[i] = rand()/(float)RAND_MAX;
  }
  //////////////////////////////////////////////////////////////////////////// 
 ////////////////////////// Initial A1 A2 A3 ////////////////////////////////
///////////////    T_IJK = [G_PQR;A1_IP,A2_JQ,A3_KR]  /////////////////////////
 /////////////////////////////////////////////////////////////////////////////
 /* reshape tensor T_IJK along different mode */
   float *G; G = (float*)malloc(P*Q*R*sd);


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
 tucker(X,G,h_A1,h_A2,h_A3,I,J,K);
 
  free(X); free(h_A1); free(h_A2); free(h_A3); free(G);

  
  return 0;
}

