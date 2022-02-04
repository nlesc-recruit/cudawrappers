

LIBS = -L../../build/ -lcuda -lnvrtc


CUDA_WRAPPER_INCLUDE = -I../../
CUDA_WRAPPER_LIB = -Wl,-rpath=../../build/ -lcu



all: vector_add


vector_add: vector_add.cc vector_add_kernel.cu
	g++ $(LIBS) $(CUDA_WRAPPER_LIB) vector_add.cc $(CUDA_WRAPPER_INCLUDE) -o vector_add


clean:
	rm vector_add
