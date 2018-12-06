#include "head.h"

using namespace std;

int main(int argc,char *argv[]){

	int a = atoi(argv[1]);
	int b = atoi(argv[2]);
	int c = atoi(argv[3]);
	int r = 2;

	dt *A = new dt[a*b*c]();
	srand(0);
	for(int i = 0;i<a*b*c;i++){
		A[i] = rand()%4;
	}
	printTensor(A,a,b,c);
	
	dt *left = new dt[c*r]();
	for(int i = 0;i<c*r;i++){
		left[i] = rand()%4;
	}
	printTensor(left,c,r,1);

	dt *right = new dt[b*r]();
	for(int i = 0;i<b*r;i++){
		right[i] = rand()%4;
	}
	printTensor(right,b,r,1);

	dt *res = new dt[a*r]();

	allin(A,left,right,res,c,b,r,a,1);
	printTensor(res,a,r,1);

	delete[] A; A = nullptr;
	delete[] left; left = nullptr;
	delete[] right; right = nullptr;
	delete[] res; res = nullptr;

	return 0;
}
