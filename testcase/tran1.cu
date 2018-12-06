#include "head.h"
using namespace std;

int main(int argc,char *argv[]){
	int r;
	int cnt[10] = {80,128,256,512,640,704,768,832,896,960};
for(int i = 0;i<10;i++){
	int a = cnt[i];
	int b = a;
	int c = a;
	if(a<10){
		r = 1;
	}else{
	 r = a/8;
	}
	dt *X = new dt[a*b*c]();
	srand(0);
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
	cp_als(X,A,B,C,a,b,c,r);
	cp_als(X,A,B,C,a,b,c,r);
	double sum = 0.0;
for(int j = 0;j<5;j++){
/*	dt *temp1 = new dt[a*r]();
	dt *temp2 = new dt[b*r]();
	dt *temp3 = new dt[c*r]();
	dt *tem1 = new dt[r*r]();
	dt *tem2 = new dt[r*r]();
	dt *tem3 = new dt[r*r]();
*/
	clock_t start,end;
	start = clock();
	for(int mei = 0;mei<1;mei++){

		allin(X,C,B,A,c,b,r,a,1);
	//	KRao(X,C,B,temp1,tem1,c,b,r,a,1);
	//	solve(tem1,temp1,A,r,a);     // we get A  

		allin(X,C,A,B,c,a,r,b,2);
	//	KRao(X,C,A,temp2,tem2,c,a,r,b,2);
	//	solve(tem2,temp2,B,r,b);     // we get B
		
		allin(X,B,A,C,b,a,r,c,3);
	//	KRao(X,B,A,temp3,tem3,b,a,r,c,3);
	//	solve(tem3,temp3,C,r,c);    //we get C

//		recontr(X,X_temp,A,B,C,a,b,c,r);

//		error[i] = 
	}
	end=clock();
	sum = sum+(double)(end-start)/CLOCKS_PER_SEC;
/*	delete[] temp1;temp1 = nullptr;
	delete[] temp2;temp1 = nullptr;
	delete[] temp3;temp1 = nullptr;
	delete[] tem1;tem1 = nullptr;
	delete[] tem2;tem2 = nullptr;
	delete[] tem3;tem3 = nullptr;
*/
	}


	ofstream outfl("ctime.txt",ios::app);
	outfl<<a<<"*"<<a<<"*"<<a<<"  ";
	outfl<<sum/5<<"s"<<endl;
	outfl.close();
	delete[] X;X = nullptr;
	delete[] A;A = nullptr;
	delete[] B;B = nullptr;
	delete[] C;C = nullptr;
//	break;
}
	return 0;

}
