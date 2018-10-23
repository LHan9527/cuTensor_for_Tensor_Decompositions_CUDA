/*************************************************************************
	> File Name: tensor.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年09月09日 星期日 20时28分47秒
 ************************************************************************/

#include "opera.h"
#include <fstream>
typedef float dt;
using namespace std;
int main(int argc,char *argv[]){
//	int a = atoi(argv[1]);
//	int b = atoi(argv[2]);
//	int c = atoi(argv[3]);		//Tensor size of a*b*c
	int r1,r2,r3,r;
//	int cnt[10] = {80,128,256,512,640,704,768,832,896,1024};
	int cnt[27] = {80,128,256,512,640,704,768,832,896,80,128,256,512,640,704,768,832,896,80,128,256,512,640,704,768,832,896};
for(int i = 0;i<27;i++){
	int	a = cnt[i];
	int b = a;
	int c = a;
	if(a<10){
		r1 = 1;
		r2 = 1;
		r3 = 1;
	}else{
		 r1 = a/8;			//assume the core size of r1*r2*r3
		 r2 = b/8;
		 r3 = c/8;
		
	}
    cout<<r1<<endl;
	dt *A = new dt[a*b*c]();   //Tensor to be decom
	dt *core = new dt[r1*r2*r3]();

	srand(0);
	for(int i = 0;i<a*b*c;i++){
		A[i] = rand()*0.1/(RAND_MAX*0.1);		//initial Tensor A
	}

//	printTensor(A,a,b,c);

	dt *U1 = new dt[a*r1]();	//a*r1
	dt *U2 = new dt[b*r2]();	//b*r2
	dt *U3 = new dt[c*r3]();	//c*r3  3 mat factors

	clock_t start,end;
	start = clock();
	HOSVD(A,core,U1,U2,U3,a,b,c);	//function for tuckey 
	end = clock();
	cout<<a<<"*"<<a<<"  ";
	cout<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
//	printTensor(core,r1,r2,r3);
	
	ofstream outfile("ttime.txt",ios::app);
	outfile<<a<<"*"<<a<<"  ";
	outfile<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
	outfile.close();
	cout<<"all is over"<<endl;


	delete[] A; A = nullptr;
	delete[] U1; U1 = nullptr;
	delete[] U2; U2 = nullptr;
	delete[] U3; U3 = nullptr;
	delete[] core; core = nullptr;
//	break;
}

for(int i = 0;i<27;i++){
    int a = cnt[i];
    int b = a;
    int c = a;
	if(a<10){
		r = 1;
	}else{
		r = a/8;
	}

	dt *X = new dt[a*b*c]();
	srand(2);
	for(int i = 0;i<a*b*c;i++){
		X[i] = rand()*0.1/(RAND_MAX*0.1);
	}
//	printTensor(X,a,b,c);
	dt *A = new dt[a*r]();
	dt *B = new dt[b*r]();
	dt *C = new dt[c*r]();
	for(int i = 0;i<a*r;i++){
		A[i] = rand()*0.1/(RAND_MAX*0.1);
	}
//	printTensor(A,a,r,1);
	for(int i = 0;i<b*r;i++){
		B[i] = rand()*0.1/(RAND_MAX*0.1);
	}
//	printTensor(B,b,r,1);
	for(int i = 0;i<c*r;i++){
		C[i] = rand()*0.1/(RAND_MAX*0.1);
	}
//	printTensor(C,c,r,1);
	clock_t start,end;
	start = clock();
	cp_als(X,A,B,C,a,b,c,r);
	end = clock();
	ofstream outfl("ctime.txt",ios::app);
    outfl<<a<<"*"<<a<<"*"<<a<<"  "; 
	outfl<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
	outfl.close();
	delete[] X;X = nullptr;
	delete[] A;A = nullptr;
	delete[] B;B = nullptr;
	delete[] C;C = nullptr;
//	break;
}
	return 0;

}
