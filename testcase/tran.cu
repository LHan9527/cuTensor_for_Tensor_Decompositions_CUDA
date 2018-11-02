#include "head.h"
using namespace std;

int main(int argc,char *argv[]){
	int a = atoi(argv[1]);
	int b = atoi(argv[2]);
	int c = atoi(argv[3]);
	//A a*b  B a*c  res b*c
	dt *A = new dt[a*b*c]();
	dt *B = new dt[a*b]();
	dt *res = new dt[a*a*c]();
	dt *res2 = new dt[a*a*c]();

	for(int i = 0;i<a*b*c;i++){
		A[i] = rand()*0.1/(RAND_MAX*0.1);
	}
//	printTensor(A,a,b,1);
	for(int i = 0;i<a*b;i++){
		B[i] = rand()*0.1/(RAND_MAX*0.1);
	}
//	printTensor(B,a,b,1);

	clock_t time1,time2,time3;
	time1 = clock();

	cuStrideModetran(A,B,res, a, b, c);
	cout<<endl;
	time2 = clock();
	cuStrideModenotran(A,B,res2, a, b, c);
	time3 = clock();

	cout<<(double)(time2-time1)/CLOCKS_PER_SEC<<"s"<<endl;
	cout<<(double)(time3-time2)/CLOCKS_PER_SEC<<"s"<<endl;
//	printTensor(res,b,c,1);
//	printTensor(res2,b,c,1);

	delete[] A;  A=nullptr;
	delete[] B; B=nullptr;
	delete[] res;  res=nullptr;
	delete[] res2;  res2=nullptr;

	return 0;

}
