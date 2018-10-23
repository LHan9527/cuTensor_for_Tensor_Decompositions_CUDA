cc = nvcc
prom = tensor
source = tensor.cpp *.cu
lib = -lcublas -lcusolver -std=c++11

$(prom):$(source)
	$(cc) -o $(prom) $(source) $(lib)

clean:
	rm -rf $(prom)
	rm -rf ttime.txt
	rm -rf ctime.txt
