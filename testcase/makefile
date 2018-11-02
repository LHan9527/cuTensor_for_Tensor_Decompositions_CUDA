cc = nvcc
prom = tran
source = tran.cu head.cu head1.cu
lib = -lcublas -std=c++11

$(prom):$(source)
	$(cc) -o $(prom) $(source) $(lib)

clean:
	rm -rf $(prom)
