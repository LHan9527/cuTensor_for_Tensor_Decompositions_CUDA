#include "head.h"
using namespace std;

int main(int argc,char *argv[]){
	int a = atoi(argv[1]);
	int b = atoi(argv[2]);
	int c = atoi(argv[3]);
	dt *A = new dt[a*b*c]();
//	int m = 2; 
//	int n = 2; 
//	int k = 2; 
	int m = 0.1*a;
	int n = 0.1*b;
	int k = 0.1*c;
	dt *res1 = new dt[m*n*k]();
	dt *res2 = new dt[m*n*k]();
	dt *res3 = new dt[m*n*k]();
	dt *U1 = new dt[a*m]();
	dt *U2 = new dt[a*m]();
	dt *U3 = new dt[c*k]();

	srand(2);
	for(int i = 0;i<a*b*c;i++){
		A[i] = rand()/RAND_MAX;
	//	A[i] = rand()%4;
	}

	for(int i = 0;i<a*m;i++){
	//	U1[i] = rand()%4;
		U1[i] = rand()/(float)RAND_MAX;
	}
	for(int i = 0;i<b*n;i++){
	//	U2[i] = rand()%4;
		U2[i] = rand()/(float)RAND_MAX;
	}
	for(int i = 0;i<c*k;i++){
	//	U3[i] = rand()%4;
		U3[i] = rand()/(float)RAND_MAX;
	}

//	printTensor(A,a,b,c);
	dt *A1 = new dt[a*b*c]();
	dt *A2 = new dt[a*b*c]();
	dt *A3 = new dt[a*b*c]();


//	printTensor(U1,a,m,1);
//	printTensor(U2,b,n,1);
//	printTensor(U3,c,k,1);
	clock_t time1,time2,time3,time4;

	time1 = clock();
	Btensor2mat(A,A1,A2,A3,a,b,c);
//	printTensor(A1,a,b*c,1);
//	printTensor(A2,b,a*c,1);
//	printTensor(A3,c,b*a,1);

	getvector1(A1,U1,a,b*c,m);
	getvector1(A2,U2,b,a*c,n);
	getvector1(A3,U3,c,b*a,k);

	newtest16(A,U1,U2,U3,res1,a,b,c);
	time2 = clock();

/*	printTensor(U1,a,2,1);
	getvector(A1,U1,a,b*c,m);
	getvector(A2,U2,b,a*c,n);
	getvector(A3,U3,c,b*a,k);
*/
//	time4 = clock();
//	printTensor(U2,a,2,1);

//	newtest(A,U1,U2,U3,res1,a,b,c);


	
//	cuStrideMode(A,U1,res2,a,b,c);
//	time3 = clock();

/*	newtest16(A,U1,U2,U3,res2,a,b,c);
	time3 = clock();
	newtest16h(A,U1,U2,U3,res3,a,b,c);
*/
	cout<<(double)(time2-time1)/CLOCKS_PER_SEC<<"s"<<endl;
//	cout<<(double)(time3-time2)/CLOCKS_PER_SEC<<"s"<<endl;
//	cout<<(double)(time4-time3)/CLOCKS_PER_SEC<<"s"<<endl;

//	printTensor(res1,m,n,k);
//	printTensor(res2,m,b,c);

	delete[] A;  A=nullptr;
	delete[] A1;  A1=nullptr;
	delete[] A2;  A2=nullptr;
	delete[] A3;  A3=nullptr;
	delete[] U1;  U1=nullptr;
	delete[] U2;  U2=nullptr;
	delete[] U3;  U3=nullptr;
	delete[] res1; res1=nullptr;
	delete[] res2; res2=nullptr;
	delete[] res3; res3=nullptr;

	return 0;

}
