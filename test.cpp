/*************************************************************************
	> File Name: test.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年09月09日 星期日 20时28分47秒
 ************************************************************************/

#include "opera.h"
int main(int argc,char *argv[]){
/*	int a = atoi(argv[1]);
	int b = atoi(argv[2]);
	int c = atoi(argv[3]);
	int d = atoi(argv[4]);

	dt *A = new dt[a*b]();
	dt *core = new dt[c*d];
	dt *result = new dt[a*b*c*d]();
	//srand((unsigned)time(NULL));
	srand(0);
	for(int i = 0;i<a*b;i++){
		A[i] = rand()%3+2;
	}
	for(int i = 0;i<c*d;i++){
		core[i] = rand()%3+2;
	}
	printTensor(A,a,b,1);
	printTensor(core,c,d,1);


	clock_t start,end;
	start = clock();

	result = KronPro(A,core,a,b,c,d);
	printTensor(result,a*c,b*d,1);
	end = clock();
	cout<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<endl;

	delete[] A;A=nullptr;
	delete[] core;core=nullptr;
	delete[] result;result=nullptr;
*/
	int a[10] = {1000,1100,1200,1300,1400,1500,1600,1700,1800,1900};
	for(int i = 0;i<10;i++){
		int m = a[i];

//		int m = atoi(argv[1]);
		dt *A = new dt[m*m]();
		srand(0);
		for(int i = 0;i<m*m;i++){
			A[i] = rand()%3+2;
		}
	//printTensor(A,m,m,1);

		dt *B = new dt[m*m]();

		clock_t start,end;
		start = clock();

		cuinv(A,B,m);
		
		end = clock();
		cout<<m<<"*"<<m<<"  ";
		cout<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
		
	//	printTensor(B,m,m,1);
		delete[] A;A = nullptr;
		delete[] B;B = nullptr;
	}

		cout<<"all is over"<<endl;

		return 0;
}
