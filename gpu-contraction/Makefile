EXEC = $(basename $(wildcard *.cpp) $(wildcard *.cu))

include Makefile.inc

# Extra dependencies for executables

test.o: INCLUDES += -I../cublas_batch_gemm
test: LDLIBS += -lcublas

tucker_decomp_kernel.o: INCLUDES += -I../cublas_batch_gemm
tucker_decomp_kernel: LDLIBS += -lcublas
