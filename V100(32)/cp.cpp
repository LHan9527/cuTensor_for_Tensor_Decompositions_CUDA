/*************************************************************************
	> File Name: cp.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年09月22日 星期六 19时48分28秒
 ************************************************************************/

#include "opera.h"

typedef float dt;
using namespace std;

int main(int argc,char *argv[]){
//	int a = atoi(argv[1]);
//	int b = atoi(argv[2]);
//	int c = atoi(argv[3]);
	//int r = a*0.1;
	int r;
	int cnt[10] = {80,128,256,512,640,704,768,832,896,1024};
for(int i = 0;i<10;i++){
	int a = cnt[i];
	int b = a;
	int c = a;
	if(a<10){
		r = 1;
	}else{
		r = a/8;
		while(r%8!=0){
			r--;
		}
	}
	cout<<r<<endl;

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
	cout<<a<<"*"<<a<<"*"<<a<<"  ";
	cout<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
/*	dt *temp1 = new dt[a*r]();
	dt *temp2 = new dt[b*r]();
	dt *temp3 = new dt[c*r]();
	dt *tem1 = new dt[r*r]();
	dt *tem2 = new dt[r*r]();
	dt *tem3 = new dt[r*r]();
	KRao(X,C,B,temp1,tem1,c,b,r,a,1);
	solve(tem1,temp1,A,r,a);
	
	KRao(X,C,A,temp2,tem2,c,a,r,b,2);
	solve(tem2,temp2,B,r,b);
	
	KRao(X,B,A,temp3,tem3,b,a,r,c,3);
	solve(tem3,temp3,C,r,c);

	printTensor(temp1,a,r,1);
	printTensor(tem1,r,r,1);
	printTensor(A,a,r,1);

	printTensor(temp2,b,r,1);
	printTensor(tem2,r,r,1);
	printTensor(B,b,r,1);
	
	printTensor(temp3,c,r,1);
	printTensor(tem3,r,r,1);
	printTensor(C,c,r,1);

	delete[] temp1;temp1 = nullptr;
	delete[] temp2;temp1 = nullptr;
	delete[] temp3;temp1 = nullptr;
	delete[] tem1;tem1 = nullptr;
	delete[] tem2;tem2 = nullptr;
	delete[] tem3;tem3 = nullptr;
*/
	delete[] X;X = nullptr;
	delete[] A;A = nullptr;
	delete[] B;B = nullptr;
	delete[] C;C = nullptr;
}
	return 0;
}
