/*************************************************************************
	> File Name: test.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年09月09日 星期日 20时28分47秒
 ************************************************************************/

#include "opera.h"
typedef float dt;
using namespace std;
int main(int argc,char *argv[]){
	int a = atoi(argv[1]);
	int b = atoi(argv[2]);
	int c = atoi(argv[3]);
	int r = atoi(argv[4]);
/*	dt *A = new dt[a*b]();
//	int r1 = 0.1*a;
//	int r2 = 0.1*b;
//	int r3 = 0.1*c;
	
	dt *B = new dt[b*c]();
	dt *C = new dt[a*c]();

	//srand((unsigned)time(NULL));
	srand(0);
	for(int i = 0;i<a*b;i++){
		A[i] = rand()%3+2;
	}
	for(int i = 0;i<b*c;i++){
		B[i] = rand()%3+2;
	}
	printTensor(A,a,b,1);
	printTensor(B,b,c,1);
	
	dt *CC = new dt[a*c]();

	clock_t start,end;
	start = clock();

	maxpro(A,B,C,a,b,c);
//	V100maxpro(A,B,CC,a,b,c);


	end = clock();
	cout<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<endl;

	printTensor(C,a,c,1);
//	printTensor(CC,a,c,1);	
*/	
	dt *AA = new dt[a*b*c]();
	srand(0);
	for(int i = 0;i<a*b*c;i++){
		AA[i] = rand()%3+2;
	}
	printTensor(AA,a,b,c);
	dt *BB = new dt[b*r]();
	for(int i = 0;i<b*r;i++){
		BB[i] = rand()%3+2;
	}
	printTensor(BB,b,r,1);
	dt *CCC = new dt[a*r*b]();
	v100mpStride(AA,BB,CCC,a,b,c,r);
	printTensor(CCC,a,r,c);

//	delete[] A; A = nullptr;
//	delete[] B; B = nullptr;
//	delete[] C; C = nullptr;
	delete[] AA; AA = nullptr;
	delete[] BB; BB = nullptr;
	delete[] CCC; CCC = nullptr;
//	delete[] CC; CC = nullptr;

	cout<<"all is over"<<endl;
	return 0;

}
