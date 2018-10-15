/*************************************************************************
	> File Name: tensor.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年09月09日 星期日 20时28分47秒
 ************************************************************************/

#include "opera.h"
typedef float dt;
using namespace std;
int main(int argc,char *argv[]){
//	int a = atoi(argv[1]);
//	int b = atoi(argv[2]);
//	int c = atoi(argv[3]);		//Tensor size of a*b*c
	int r1,r2,r3;
	int cnt[10] = {80,128,256,512,640,704,768,832,896,1024};
for(int i = 0;i<10;i++){
	int a = cnt[i];
	int b = a;
	int c = a;
	if(a<10){
		r1 = 1;
		r2 = 1;
		r3 = 1;
	}else{
		 r1 = a/8;			//assume the core size of r1*r2*r3
		//int r1 = 2;			//assume the core size of r1*r2*r3
		 r2 = b/8;
		 r3 = c/8;
		while(r1%8!=0){
			r1--;
		}
		while(r2%8!=0){
			r2--;
		}
		while(r3%8!=0){
			r3--;
		}
		//int r2 = 3;
		//int r3 = 2;
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
	cout<<a<<"*"<<a<<"*"<<a<<"  ";
	cout<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
//	printTensor(core,r1,r2,r3);
	cout<<"all is over"<<endl;


	delete[] A; A = nullptr;
	delete[] U1; U1 = nullptr;
	delete[] U2; U2 = nullptr;
	delete[] U3; U3 = nullptr;
	delete[] core; core = nullptr;
}
	return 0;

}
