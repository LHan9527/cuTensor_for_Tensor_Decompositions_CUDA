# cuTensor_for_Decompositions_CUDA

In testcase,which contains some test functions and my new impletion of 2 tensor decompletion.So, you can only see it and ignore others.

BTAS matlab gpu-contraction are 3 way of implement tucker decomposition(HOSVD) which modified from https://github.com/shiyangdaisy23/tensor-contraction.


 Goto testcase run tran.cu and tran1.cu to check running time of 2 decomposition.

contribution:

1 the use of tensor core.

2 In tucker,when deal with G = X×1U1' ×2U2' ×3U3',I use 3 tensor contracton to solve

3 In cp,I use QR decompletion to get the A without computing Moore–Penrose pseudoinverse.Also,in cublas data store in column,we exploit it to solve transpose operation.

4 Symmetric matrix will appear when decomsition .So we store with Triangle .


The running time of Tucker decomposition is shown as follows


![tucker time](https://github.com/hust512/cuTensor_for_Decompositions_CUDA/blob/master/curve/tucker.jpg)

The running time of CP decomposition is shown as follows


![cp time](https://github.com/hust512/cuTensor_for_Decompositions_CUDA/blob/master/curve/cp.jpg)

The speedup of Tucker decomposition is shown as follows


![tucker time](https://github.com/hust512/cuTensor_for_Decompositions_CUDA/blob/master/curve/tspeedup.jpg)

The speedup of CP decomposition is shown as follows


![cp time](https://github.com/hust512/cuTensor_for_Decompositions_CUDA/blob/master/curve/cspeedup.jpg)
    
