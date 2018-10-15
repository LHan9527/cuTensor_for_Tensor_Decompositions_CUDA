# cuTensor_for_Decompositions_CUDA
Tensor decompositions on CUDA.And we implement two methods on general GPU and V100(with tensor core)respectively.On V100,We choose half float and float as input,and get two result. 

1. run tensor.cpp to check running time of 2 decomposition.

2. Goto V100(16) and V100(32) to test 2 kinds of precision.


The running time of Tucker decomposition is shown as follows


![tucker time](https://github.com/hust512/cuTensor_for_Decompositions_CUDA/blob/master/curve/tucker.jpg)

The running time of CP decomposition is shown as follows


![cp time](https://github.com/hust512/cuTensor_for_Decompositions_CUDA/blob/master/curve/cp.jpg)

The speedup of Tucker decomposition is shown as follows


![tucker time](https://github.com/hust512/cuTensor_for_Decompositions_CUDA/blob/master/curve/tspeedup.jpg)

The speedup of CP decomposition is shown as follows


![cp time](https://github.com/hust512/cuTensor_for_Decompositions_CUDA/blob/master/curve/cspeedup.jpg)
    
