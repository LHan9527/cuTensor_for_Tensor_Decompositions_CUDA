#include "head.h"
using namespace std;

int main(int argc,char *argv[]){
	int m,n,k;
//	int a = atoi(argv[1]);
//	int b = atoi(argv[2]);
//	int c = atoi(argv[3]);
	int cnt[10] = {80,128,256,512,640,704,768,832,896,960};
for(int i = 0;i<10;i++){
//	int m = 2; 
//	int n = 2; 
//	int k = 2; 
	int a = cnt[i];
	int b = a;
	int c = a;
	if(a<10){
		m = 1;
		 n=1;
		 k=1;
	}else{
	 m = a/8;
	 n = b/8;
	 k = c/8;}
	dt *A = new dt[a*b*c]();
	dt *res1 = new dt[m*n*k]();
	dt *U1 = new dt[a*m]();
	dt *U2 = new dt[a*m]();
	dt *U3 = new dt[c*k]();

	srand(0);
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


//	printTensor(U1,a,m,1);
//	printTensor(U2,b,n,1);
//	printTensor(U3,c,k,1);
	Hosvd(A,res1,U1,U2,U3,a,b,c);
	Hosvd(A,res1,U1,U2,U3,a,b,c);
	double sum = 0.0;
for(int j = 0;j<5;j++){
	if(a<10){
		 m = 1;
		 n=1;
		 k=1;
	}else{
	 m = a/8;
	 n = b/8;
	 k = c/8;}

	clock_t tt1,tt2;
	tt1=clock();
	dt *A1 = new dt[a*b*c]();
	dt *A2 = new dt[a*b*c]();
	dt *A3 = new dt[a*b*c]();
	Btensor2mat(A,A1,A2,A3,a,b,c);
	getvector1(A1,U1,a,b*c,m);
	getvector1(A2,U2,b,a*c,n);
	getvector1(A3,U3,c,b*a,k);
	newtest16(A,U1,U2,U3,res1,a,b,c);
	
	delete[] A1;  A1=nullptr;
	delete[] A2;  A2=nullptr;
	delete[] A3;  A3=nullptr;
	tt2=clock();
//	cout<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<endl;
	sum = sum+(double)(tt2-tt1)/CLOCKS_PER_SEC;
}
	ofstream outfile("ttime.txt",ios::app);
	outfile<<a<<"*"<<a<<"*"<<a<<"  ";
	outfile<<sum/5<<"s"<<endl;
	outfile.close();
	cout<<"all is over"<<endl;

//	printTensor(A1,a,b*c,1);
//	printTensor(A2,b,a*c,1);
//	printTensor(A3,c,b*a,1);

//	getvector1(A1,U1,a,b*c,m);
//	getvector1(A2,U2,b,a*c,n);
//	getvector1(A3,U3,c,b*a,k);

//	newtest16(A,U1,U2,U3,res1,a,b,c);
/*	printTensor(U1,a,2,1);
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
//	cout<<(double)(time3-time2)/CLOCKS_PER_SEC<<"s"<<endl;
//	cout<<(double)(time4-time3)/CLOCKS_PER_SEC<<"s"<<endl;

//	printTensor(res1,m,n,k);
//	printTensor(res2,m,b,c);

	delete[] A;  A=nullptr;
	delete[] U1;  U1=nullptr;
	delete[] U2;  U2=nullptr;
	delete[] U3;  U3=nullptr;
	delete[] res1; res1=nullptr;
}
	return 0;

}
